#!/usr/bin/env python3
"""
eStudio Video Production Workflow v3 - Quality-First Pipeline

New in v3 (vs v2):
  - WanVideo 2.2 I2V A14B (better motion + character consistency)
  - GPT-5 (full) for higher-quality script/storyboard
  - Multi-LLM Director: Writer + DOP + Editor (STEP 2, inspired by LTX LLM Directors)
  - Character memory buffer: 5 reference images batched (StoryMem-inspired)
  - Scene chaining: last-frame extractor for optional next-scene bootstrap (Sora2Alike)
  - Higher-quality sampling: SDXL 40 steps, WanVideo 30 steps
  - Compatibility mode: no IPAdapter / RIFE / Whisper-subtitle / ChatterBox dependency

Pipeline:
  STEP 0: Load shared models (Wan 2.2)
  STEP 1: LLM generates script from concept (GPT-5)
  STEP 2: Storyboard + prompts + Multi-LLM Director (Writer/DOP/Editor)
  STEP 3: Character memory buffer (5 refs) + style ref
  STEP 4+5: Merged loop - image gen + WanVideo I2V per scene + last-frame extract
  STEP 6: Voice (ElevenLabs) + SRT subtitles text
  STEP 7: Dynamic assembly (raw/final video)

Usage: OPENAI_API_KEY=sk-xxx python generate_video_workflow_v3.py > workflow-video-production-v3.json
"""
import json, uuid, os

_nid = 0; _lid = 0; nodes = []; links = []

def nid():
    global _nid; _nid += 1; return _nid
def lid():
    global _lid; _lid += 1; return _lid

def N(t, pos, sz, w=None, title=None, ins=None, outs=None, color=None, bgcolor=None, mode=0, props=None):
    n = {"id":nid(),"type":t,"pos":pos,"size":sz,"flags":{},"order":0,"mode":mode,
         "inputs":[],"outputs":[],"properties":props or {},"widgets_values":w or []}
    if title: n["title"] = title
    if color: n["color"] = color
    if bgcolor: n["bgcolor"] = bgcolor
    for inp in (ins or []):
        s = {"name":inp[0],"type":inp[1],"link":None}
        if len(inp)>2: s["shape"]=inp[2]
        n["inputs"].append(s)
    for i,out in enumerate(outs or []):
        n["outputs"].append({"name":out[0],"type":out[1],"links":[],"slot_index":i})
    nodes.append(n); return n["id"]

def C(src, ss, dst, ds, t="*"):
    l = lid(); links.append([l,src,ss,dst,ds,t])
    for n in nodes:
        if n["id"]==src and ss<len(n["outputs"]): n["outputs"][ss]["links"].append(l)
        if n["id"]==dst and ds<len(n["inputs"]): n["inputs"][ds]["link"]=l
    return l

groups = []

# =====================================================================
# STEP 0: SHARED MODELS & CONFIG
# =====================================================================
X0, Y0 = 0, 0

# --- Image generation models ---
n_ckpt = N("CheckpointLoaderSimple",[X0,Y0],[315,98],
    w=["realvisxlV50_v50LightningBakedvae.safetensors"],
    title="SDXL Checkpoint",
    outs=[("MODEL","MODEL"),("CLIP","CLIP"),("VAE","VAE")])

n_lora = N("LoraLoader",[X0,Y0+130],[315,130],
    w=["cinematic_v1.safetensors",0.7,0.7],
    title="Style LoRA (Cinematic)",
    ins=[("model","MODEL"),("clip","CLIP")],
    outs=[("MODEL","MODEL"),("CLIP","CLIP")])
C(n_ckpt,0,n_lora,0,"MODEL")
C(n_ckpt,1,n_lora,1,"CLIP")

# --- WanVideo models ---
n_wanmod = N("WanVideoModelLoader",[X0+400,Y0],[400,150],
    w=["wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors","bf16","fp8_e4m3fn","offload_device","sdpa"],
    title="WanVideo 2.2 Model (I2V 14B high-noise)",
    ins=[("compile_args","WANCOMPILEARGS",7),("block_swap_args","BLOCKSWAPARGS",7),
         ("lora","WANVIDLORA",7),("vram_management_args","VRAM_MANAGEMENTARGS",7),
         ("extra_model","VACEPATH",7),("fantasytalking_model","FANTASYTALKINGMODEL",7),
         ("multitalk_model","MULTITALKMODEL",7),("fantasyportrait_model","FANTASYPORTRAITMODEL",7),
         ("rms_norm_function","COMBO",7)],
    outs=[("model","WANVIDEOMODEL")])

n_wanvae = N("WanVideoVAELoader",[X0+400,Y0+180],[315,100],
    w=["wan2.2_vae.safetensors","bf16",False,False],
    title="WanVideo 2.2 VAE",
    ins=[("compile_args","WANCOMPILEARGS",7)],
    outs=[("vae","WANVAE")])

n_clipv = N("CLIPVisionLoader",[X0+400,Y0+270],[315,58],
    w=["clip_vision_h.safetensors"],title="CLIP Vision",
    outs=[("CLIP_VISION","CLIP_VISION")])

# --- LLM Config (API key from environment) ---
n_llm = N("DirectLLMConfig",[X0+400,Y0+370],[400,130],
    w=[os.environ.get("OPENAI_API_KEY",""),
       "https://api.openai.com/v1","gpt-5"],
    title="LLM Config (GPT-5 full)",
    outs=[("config","LLM_CONFIG")])

# --- Global params ---
n_w = N("easy int",[X0+850,Y0],[210,58],w=[480],title="Width",outs=[("INT","INT")])
n_h = N("easy int",[X0+850,Y0+70],[210,58],w=[832],title="Height",outs=[("INT","INT")])
n_nf = N("easy int",[X0+850,Y0+140],[210,58],w=[49],title="Num Frames (WanVideo ~2s before RIFE)",outs=[("INT","INT")])
# FPS=24 hardcoded in VHS_VideoCombine widgets

groups.append({"title":"STEP 0: SHARED MODELS & CONFIG","bounding":[X0-30,Y0-60,1500,620],"color":"#A88"})

# =====================================================================
# STEP 1: SCRIPT GENERATION (3-stage pipeline)
#   1A: Concept Input + Video Parameters
#   1B: Story Structure (hook, climax, ending, storyline arc)
#   1C: Full Script with scene breakdown
# =====================================================================
X1, Y1 = 1800, 0

# --- 1A: Concept Input ---
n_concept = N("Text Multiline",[X1,Y1-200],[500,200],
    w=["Video ngan day tieng Nhat duong pho voi linh vat HeyJ (cho Shiba vang, 2D flat cartoon).\nChu de: Japanese slang to sound like a local\nSo canh: 11 (Hook + Setup + 7 tu vung + Ending + CTA)\n\nKich ban chi tiet:\n1. HOOK: Shibi cool ngau deo kinh | Voice: Japanese slang to sound like a local\n2. SETUP: Shibi chi vao camera | Voice: Stop sounding like a textbook... say this instead\n3. Shibi lo tay lam roi do, cuoi | Tu: てへぺろ (tehepero) = oops | Voice\n4. Shibi buc | Tu: キレる (kireru) = lose it / snap | Voice\n5. Shibi nhay nhot | Tu: パリピ (paripi) = party people | Voice\n6. Shibi confused | Tu: わかんない (wakannai) = I dont get it | Voice\n7. Shibi cuoi lon | Tu: ウケる (ukeru) = thats funny | Voice\n8. Shibi che | Tu: ダサい (dasai) = lame / cringe | Voice\n9. Shibi bat ngo | Tu: マジ (maji) = for real? | Voice\n10. ENDING: Shibi cool | Voice: Now you sound less like a textbook\n11. CTA: Shibi chi xuong | Text: Follow Shibi for real Japanese | Voice off"],
    title="CONCEPT INPUT",
    outs=[("STRING","STRING")])

n_target = N("Text Multiline",[X1,Y1+50],[500,130],
    w=["Doi tuong: Gen Z, nguoi hoc tieng Nhat, fan anime\nNen tang: TikTok / YouTube Shorts / Instagram Reels (9:16 doc)\nTone: vui nhon, nhanh, hoc ma vui, meme-friendly\nPhong cach: 2D flat cartoon mascot + text overlay lon ro rang\nMuc tieu: nguoi xem nho duoc 2-3 tu tieng Nhat, follow de hoc tiep"],
    title="TARGET AUDIENCE & TONE",
    outs=[("STRING","STRING")])

n_duration = N("Text Multiline",[X1,Y1+230],[500,80],
    w=["Thoi luong: 45-60 giay | So canh: 11 | FPS: 24 | Aspect: 9:16 (1080x1920)\nMoi canh noi dung: 2-4 giay | Hook: 2 giay | CTA: 3 giay"],
    title="VIDEO SPECS",
    outs=[("STRING","STRING")])

# Merge all inputs for context
n_concat1a = N("Text Concatenate",[X1+550,Y1-50],[300,120],
    w=["\\n---\\n","true"],
    title="Merge Inputs",
    ins=[("delimiter","STRING"),("clean_whitespace","COMBO"),
         ("text_a","STRING",7),("text_b","STRING",7),("text_c","STRING",7)],
    outs=[("STRING","STRING")])
C(n_concept,0,n_concat1a,2,"STRING")
C(n_target,0,n_concat1a,3,"STRING")
C(n_duration,0,n_concat1a,4,"STRING")

# --- 1B: Story Structure ---
n_structure = N("DirectLLMChat",[X1+900,Y1-200],[550,350],
    w=["Ban la chuyen gia lam video ngan viral tren TikTok/Shorts. Phan tich concept va xay dung CAU TRUC VIDEO.\n\nVideo giao duc ngan theo pattern: HOOK -> SETUP -> NOI DUNG (loop) -> ENDING -> CTA\n\nTra ve CHINH XAC:\n\n[HOOK] (1-2 giay)\nHinh anh: nhan vat lam gi, bieu cam nao\nText tren man hinh: dong chu gay chu y\nMuc tieu: dung scroll, tao to mo\n\n[SETUP] (2-3 giay)\nHinh anh: nhan vat hanh dong gi\nVoice/Text: cau dan dat vao noi dung chinh\n\n[NOI DUNG] (loop moi item 2-4 giay)\nMoi item: nhan vat pose/bieu cam + tu vung/kien thuc + text overlay\nNhip do: nhanh, cat canh gon, khong keo dai\n\n[ENDING] (2-3 giay)\nHinh anh: nhan vat ket\nVoice/Text: cau ket dong lai noi dung\n\n[CTA] (2-3 giay)\nHinh anh: nhan vat chi xuong/vay tay\nText: Follow + ten kenh\nVoice: tat hoac nhe\n\n[NHIP DO TONG]\nToc do cat: nhanh (2-4 giay/canh)\nNhac nen: vui, upbeat, co beat ro\nChuyen canh: cut nhanh, bounce, zoom",
       "Phan tich cau truc video ngan:","",16384,0.8],
    title="LLM: Story Structure",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_structure,0,"LLM_CONFIG")
C(n_concat1a,0,n_structure,3,"STRING")

n_show1b = N("easy showAnything",[X1+1500,Y1-200],[280,80],w=["False"],
    title="Preview Structure",ins=[("anything","*",7)])
C(n_structure,0,n_show1b,0,"STRING")

# --- 1C: Full Script with Scene Breakdown ---
# Merge structure + original inputs for full context
n_concat1c = N("Text Concatenate",[X1+900,Y1+200],[300,120],
    w=["\\n===\\n","true"],
    title="Merge for Script",
    ins=[("delimiter","STRING"),("clean_whitespace","COMBO"),
         ("text_a","STRING",7),("text_b","STRING",7)],
    outs=[("STRING","STRING")])
C(n_concat1a,0,n_concat1c,2,"STRING")
C(n_structure,0,n_concat1c,3,"STRING")

n_script = N("DirectLLMChat",[X1+1250,Y1+200],[550,380],
    w=["Ban la dao dien video ngan chuyen nghiep. Viet KICH BAN HOAN CHINH dang bang.\n\nFormat CHINH XAC cho MOI CANH (1 dong/canh, cac cot ngan cach bang | ):\nscene_num | role | visual | noi_dung | voice | text_on_screen | duration\n\nGiai thich cot:\n- scene_num: so thu tu (1,2,3...)\n- role: hook / setup / content / ending / cta\n- visual: mo ta CHINH XAC nhan vat dang lam gi, bieu cam, tu the, phu kien (VD: Shibi cool deo kinh, Shibi buc tuc do mat, Shibi nhay nhot vui ve)\n- noi_dung: noi dung hoc/thong tin trong canh (VD: tu vung, y nghia, vi du)\n- voice: loi doc/narration cua canh (hoac \"voice off\" neu khong co)\n- text_on_screen: CHU hien tren man hinh (tu vung lon + nghia nho ben duoi)\n- duration: thoi luong giay (2-4s cho content, 1-2s cho hook)\n\nQuy tac:\n- Hook: 1-2 canh dau, bat mat ngay\n- Setup: 1 canh gioi thieu chu de\n- Content: cac canh noi dung chinh (moi canh = 1 don vi kien thuc)\n- Ending: 1 canh ket\n- CTA: 1 canh cuoi keu goi follow\n- Tong thoi luong 45-60 giay\n- Chi viet bang, KHONG giai thich gi them",
       "Viet kich ban dang bang:","",16384,0.7],
    title="LLM: Full Script",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_script,0,"LLM_CONFIG")
C(n_concat1c,0,n_script,3,"STRING")

n_show1c = N("easy showAnything",[X1+1850,Y1+200],[280,80],w=["False"],
    title="Preview Full Script",ins=[("anything","*",7)])
C(n_script,0,n_show1c,0,"STRING")

groups.append({"title":"STEP 1: SCRIPT (Concept -> Structure -> Full Script)","bounding":[X1-30,Y1-280,2200,750],"color":"#3f789e"})

# =====================================================================
# STEP 2: STORYBOARD + PROMPT SYSTEM (4-stage pipeline + DirectorsCinemaPrompt)
#   2A: Visual Storyboard (composition, framing, color per scene)
#   2B: START frame prompts from storyboard
#   2C: END frame prompts from storyboard
#   2D: Motion prompts for WanVideo (per-scene camera/movement)
#   2E: DirectorsCinemaPrompt (optional cinema preset enhancement)
# =====================================================================
X2, Y2 = 4200, 0

# --- 2A: Big Prompt 1 - Character Pose Sheet ---
# Tu kich ban, liet ke TAT CA cac pose/bieu cam can thiet cua nhan vat
n_storyboard = N("DirectLLMChat",[X2,Y2-250],[550,380],
    w=["Tu kich ban video, tao BANG POSE NHAN VAT (character pose sheet).\n\nVoi MOI CANH trong kich ban, xac dinh:\n- Nhan vat dang lam gi (tu the cu the)\n- Bieu cam mat (vui/buon/gian/ngac nhien/cool...)\n- Phu kien dac biet neu co (kinh, mu, khan...)\n- Huong nhin (nhin camera, nhin sang, nhin xuong...)\n\nFormat MOI DONG (1 canh = 1 dong):\nscene_num | pose: [tu the chi tiet] | expression: [bieu cam] | props: [phu kien] | facing: [huong nhin]\n\nVi du:\n1 | pose: standing confidently with arms crossed | expression: cool smirk with sunglasses | props: round sunglasses | facing: looking at camera\n2 | pose: pointing index finger at camera | expression: excited open mouth smile | props: none | facing: directly at camera\n3 | pose: both hands up oops gesture, dropped item on floor | expression: embarrassed tongue out smile | props: none | facing: looking at camera sheepishly\n\nSO DONG = SO CANH. Chi bang, khong giai thich.",
       "Tao bang pose nhan vat:","",16384,0.6],
    title="LLM: Character Poses",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_storyboard,0,"LLM_CONFIG")
C(n_script,0,n_storyboard,3,"STRING")

n_show_sb = N("easy showAnything",[X2+600,Y2-250],[280,80],w=["False"],
    title="Preview Storyboard",ins=[("anything","*",7)])
C(n_storyboard,0,n_show_sb,0,"STRING")

# Merge script + storyboard for prompt generation context
n_concat2 = N("Text Concatenate",[X2+600,Y2-140],[300,100],
    w=["\\n===STORYBOARD===\\n","true"],
    title="Script + Storyboard",
    ins=[("delimiter","STRING"),("clean_whitespace","COMBO"),
         ("text_a","STRING",7),("text_b","STRING",7)],
    outs=[("STRING","STRING")])
C(n_script,0,n_concat2,2,"STRING")
C(n_storyboard,0,n_concat2,3,"STRING")

# --- 2B: START frame prompts ---
n_prompt_start = N("DirectLLMChat",[X2,Y2+200],[550,320],
    w=["Tu kich ban + bang pose nhan vat, tao prompt TIENG ANH de tao ANH cho MOI CANH.\n\nRANG BUOC BAT BUOC:\n- Nhan vat LUON la cung mot mascot Shiba vang 2D flat cartoon, ti le dang tron gon, cream muzzle + cream belly, orange cheeks, black oval eyes, pink tongue, outline day sach.\n- Khong duoc bien thanh nguoi that, ban than nguoi, realistic skin, realistic hair, 3D render, anime human, doll, UI, app screen, poster, typography, watermark.\n- Moi canh phai la full body mascot, nam gon trong frame, khong crop dau, tai, tay, chan.\n- Nen don gian, sach, it chi tiet, de trong vung negative space cho subtitle.\n- Bo cuc portrait 9:16, mascot o giua khung hinh, doc ro ngay tren mobile.\n- Moi canh chi thay doi pose, expression, props, va mau background theo kich ban.\n\nFormat moi dong (1 canh = 1 dong, KHONG xuong dong):\nA clean 2D flat vector cartoon illustration of the same cute yellow Shiba mascot with cream muzzle and belly, orange cheeks, black oval eyes, pink tongue, thick clean outline, full body fully visible, centered in frame, [POSE from pose sheet], [EXPRESSION], [PROPS if any], simple pastel background with clear negative space for subtitles, portrait 9:16 composition, mobile-first framing, crisp edges, mascot-brand consistency, no text\n\nSO DONG = SO CANH. Chi prompt, khong giai thich.",
       "Tao image prompt cho moi canh:","",16384,0.5],
    title="LLM: Scene Image Prompts",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_prompt_start,0,"LLM_CONFIG")
C(n_concat2,0,n_prompt_start,3,"STRING")

# --- 2C: END frame prompts ---
n_prompt_end = N("DirectLLMChat",[X2,Y2+570],[550,320],
    w=["Tu kich ban + bang pose, tao prompt anh CUOI CANH (END frame) cho moi canh.\n\nRANG BUOC BAT BUOC:\n- Giu NGUYEN mascot Shiba vang 2D flat cartoon nhu start frame: cream muzzle + cream belly, orange cheeks, black oval eyes, pink tongue, outline day sach.\n- Full body mascot phai nam gon trong frame, centered, portrait 9:16, khong crop bat ky bo phan nao.\n- Khong tao nguoi that, realistic anatomy, UI, text, poster, watermark, speech bubble.\n- END frame chi duoc thay doi NHE so voi START frame: nhich pose, doi expression, them mot it secondary action, giu cung background family.\n- Muc tieu la tao cap keyframe de WanVideo noi chuyen dong mem, de doc, on dinh.\n\nFormat moi dong (1 canh = 1 dong, KHONG xuong dong):\nA clean 2D flat vector cartoon illustration of the same cute yellow Shiba mascot with cream muzzle and belly, orange cheeks, black oval eyes, pink tongue, thick clean outline, full body fully visible, centered in frame, slight progression from the start pose, [EXPRESSION], [PROPS if any], simple pastel background matching the scene with subtle variation, portrait 9:16 composition, crisp edges, no text\n\nSO DONG = SO CANH. Chi prompt, khong giai thich.",
       "Tao END frame prompts:","",16384,0.5],
    title="LLM: End Frame Prompts",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_prompt_end,0,"LLM_CONFIG")
C(n_concat2,0,n_prompt_end,3,"STRING")

# --- 2D: Motion/Camera prompts for WanVideo ---
n_prompt_motion = N("DirectLLMChat",[X2,Y2+940],[550,380],
    w=["Tao MOTION PROMPT tieng Anh cho tung canh video ngan.\nDay la video fast-cut TikTok style, moi canh 2-4 giay, nhung uu tien chat luong mascot 2D on dinh.\n\nRANG BUOC:\n- Chuyen dong ngan, de doc, khong qua phuc tap.\n- Tranh camera move cuc manh, whip pan, xoay camera, blur manh, bien dang nhan vat.\n- Uu tien subtle body motion + nhe camera push/pull de mascot giu dung shape.\n\nMoi dong (1 canh = 1 dong):\n[character motion] | [camera] | [speed] | [effect]\n\nSO DONG = SO CANH. Chi prompt, khong giai thich.",
       "Tao motion prompts:","",16384,0.5],
    title="LLM: Motion Prompts",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_prompt_motion,0,"LLM_CONFIG")
C(n_concat2,0,n_prompt_motion,3,"STRING")

# --- 2E: Multi-LLM Director (Writer + DOP + Editor, LTX-inspired) ---
# 2E-1: DOP (Director of Photography) - camera/lighting per scene
n_dop = N("DirectLLMChat",[X2,Y2+1310],[550,320],
    w=["Ban la DOP (Director of Photography). Tu kich ban + storyboard, viet CAMERA BIBLE cho toan bo video.\n\nTra ve bang 1 dong/canh, cach nhau bang |:\nscene_num | shot_type | lens_mm | camera_move | angle | lighting | color_temp\n\nGiai thich:\n- shot_type: wide/medium/close/extreme close\n- lens_mm: 24/35/50/85 (dung cho kich thich mascot-friendly)\n- camera_move: static/slow push/slow pull/slight dolly/handheld\n- angle: eye/low/high/dutch (it dung cho mascot)\n- lighting: soft-key/rim/bounce/golden/flat studio\n- color_temp: warm 3200K / neutral 5600K / cool 6500K\n\nUu tien static hoac subtle push de giu mascot shape. SO DONG = SO CANH. Chi bang, khong giai thich.",
       "DOP camera bible:","",16384,0.5],
    title="LLM: DOP (Cinematography)",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_dop,0,"LLM_CONFIG")
C(n_concat2,0,n_dop,3,"STRING")

n_show_dop = N("easy showAnything",[X2+600,Y2+1310],[280,80],w=["False"],
    title="Preview DOP",ins=[("anything","*",7)])
C(n_dop,0,n_show_dop,0,"STRING")

# 2E-2: Editor - pacing + transitions
n_editor = N("DirectLLMChat",[X2,Y2+1680],[550,320],
    w=["Ban la EDITOR video ngan. Tu kich ban + DOP, viet EDIT PLAN.\n\nBang 1 dong/canh, cach nhau bang |:\nscene_num | duration_s | in_point | out_point | transition_in | transition_out | beat_sync\n\nQuy tac:\n- duration_s: 2-4s content, 1-2s hook, 3s CTA\n- in_point: 0 (day la I2V, moi canh bat dau tu START frame)\n- out_point: duration (END frame)\n- transition_in/out: cut / match-cut / whip / bounce / zoom (match voi beat nhac)\n- beat_sync: on-beat / off-beat / half-beat (uu tien on-beat cho nhip nhanh)\n\nTong thoi luong ~60s. SO DONG = SO CANH.",
       "Edit plan:","",16384,0.5],
    title="LLM: Editor (Pacing)",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_editor,0,"LLM_CONFIG")
C(n_concat2,0,n_editor,3,"STRING")

n_show_editor = N("easy showAnything",[X2+600,Y2+1680],[280,80],w=["False"],
    title="Preview Editor",ins=[("anything","*",7)])
C(n_editor,0,n_show_editor,0,"STRING")

# 2E-3: DirectorsCinemaPrompt (enabled, fed by DOP context)
n_dc_prompt = N("DirectorsCinemaPrompt",[X2+950,Y2+1500],[350,250],
    w=["","None","flux_dev","cinematic","eye_level","soft-key",""],
    title="Directors Cinema Prompt",
    ins=[("subject","STRING"),("preset","STRING"),("target_model","STRING"),
         ("mood","STRING"),("camera_angle","STRING"),("lighting","STRING"),
         ("negative_prompt","STRING")],
    outs=[("positive_prompt","STRING"),("negative_prompt","STRING")],
    mode=4)  # muted by default - enable after verifying DOP output format

# --- Scene count ---
n_scene_count = N("LineCountNode",[X2+600,Y2+200],[250,80],w=["",True],
    title="Scene Count",
    ins=[("text","STRING")],outs=[("line_count","INT"),("summary","STRING")])
C(n_prompt_start,0,n_scene_count,0,"STRING")

# --- Previews ---
n_show2a = N("easy showAnything",[X2+600,Y2+300],[250,80],w=["False"],
    title="Preview Start Prompts",ins=[("anything","*",7)])
C(n_prompt_start,0,n_show2a,0,"STRING")

n_show2b = N("easy showAnything",[X2+600,Y2+570],[250,80],w=["False"],
    title="Preview End Prompts",ins=[("anything","*",7)])
C(n_prompt_end,0,n_show2b,0,"STRING")

n_show2c = N("easy showAnything",[X2+600,Y2+940],[250,80],w=["False"],
    title="Preview Motion Prompts",ins=[("anything","*",7)])
C(n_prompt_motion,0,n_show2c,0,"STRING")

groups.append({"title":"STEP 2: STORYBOARD + PROMPTS + MULTI-LLM DIRECTOR (Writer/DOP/Editor)","bounding":[X2-230,Y2-330,1600,2100],"color":"#88A"})

# =====================================================================
# STEP 3: REFERENCE SHEETS (Character + Style)
#   3A: Character ref - multi-view (front + 3/4) for IPAdapter
#   3B: Style ref - mood/color/lighting reference for consistency
#   3C: LLM negative prompt per scene type (hook vs calm vs climax)
# =====================================================================
X3, Y3 = 5600, -1200

# --- 3A: Character memory buffer (5 refs, StoryMem-inspired) ---
# Load mascot images directly - exact character consistency across all scenes
n_vd_char = N("LoadImage",[X3,Y3],[315,320],
    w=["heyj_linh_vat/1-2.png","image"],
    title="HeyJ Mascot (Ref 1 - Wave)",
    outs=[("IMAGE","IMAGE"),("MASK","MASK")])

n_char_ref2 = N("LoadImage",[X3+350,Y3],[315,320],
    w=["heyj_linh_vat/1-3.png","image"],
    title="HeyJ Mascot (Ref 2 - Jump)",
    outs=[("IMAGE","IMAGE"),("MASK","MASK")])

n_char_ref3 = N("LoadImage",[X3+700,Y3],[315,320],
    w=["heyj_linh_vat/1-4.png","image"],
    title="HeyJ Mascot (Ref 3 - Hat)",
    outs=[("IMAGE","IMAGE"),("MASK","MASK")])

n_char_ref4 = N("LoadImage",[X3,Y3+340],[315,320],
    w=["heyj_linh_vat/1-5.png","image"],
    title="HeyJ Mascot (Ref 4 - Side)",
    outs=[("IMAGE","IMAGE"),("MASK","MASK")])

n_char_ref5 = N("LoadImage",[X3+350,Y3+340],[315,320],
    w=["heyj_linh_vat/1-6.png","image"],
    title="HeyJ Mascot (Ref 5 - Front)",
    outs=[("IMAGE","IMAGE"),("MASK","MASK")])

# Memory buffer: batch 5 refs progressively for stronger IPAdapter identity lock
n_char_batch1 = N("ImageBatch",[X3+1050,Y3],[210,50],
    title="MemBuf 1+2",
    ins=[("image1","IMAGE"),("image2","IMAGE")],
    outs=[("IMAGE","IMAGE")])
C(n_vd_char,0,n_char_batch1,0,"IMAGE")
C(n_char_ref2,0,n_char_batch1,1,"IMAGE")

n_char_batch2 = N("ImageBatch",[X3+1050,Y3+70],[210,50],
    title="MemBuf +3",
    ins=[("image1","IMAGE"),("image2","IMAGE")],
    outs=[("IMAGE","IMAGE")])
C(n_char_batch1,0,n_char_batch2,0,"IMAGE")
C(n_char_ref3,0,n_char_batch2,1,"IMAGE")

n_char_batch3 = N("ImageBatch",[X3+1050,Y3+140],[210,50],
    title="MemBuf +4",
    ins=[("image1","IMAGE"),("image2","IMAGE")],
    outs=[("IMAGE","IMAGE")])
C(n_char_batch2,0,n_char_batch3,0,"IMAGE")
C(n_char_ref4,0,n_char_batch3,1,"IMAGE")

n_char_batch = N("ImageBatch",[X3+1050,Y3+210],[210,50],
    title="MemBuf +5 (Final)",
    ins=[("image1","IMAGE"),("image2","IMAGE")],
    outs=[("IMAGE","IMAGE")])
C(n_char_batch3,0,n_char_batch,0,"IMAGE")
C(n_char_ref5,0,n_char_batch,1,"IMAGE")

n_pv_char = N("PreviewImage",[X3+1050,Y3+280],[210,50],
    title="Preview Memory Buffer",ins=[("images","IMAGE")])
C(n_char_batch,0,n_pv_char,0,"IMAGE")

# --- 3B: Style reference (color/mood/lighting board) ---
n_styleref = N("DirectLLMChat",[X3,Y3+380],[500,230],
    w=["(disabled in v3.1)","(disabled)","",512,0.1],
    title="LLM: Style Prompt (disabled)",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")],
    mode=4)  # muted
C(n_llm,0,n_styleref,0,"LLM_CONFIG")
C(n_storyboard,0,n_styleref,3,"STRING")

n_clippos_style = N("CLIPTextEncode",[X3+550,Y3+380],[400,80],w=[""],
    title="Style Positive",
    ins=[("clip","CLIP"),("text","STRING")],outs=[("CONDITIONING","CONDITIONING")])
C(n_lora,1,n_clippos_style,0,"CLIP")
C(n_styleref,0,n_clippos_style,1,"STRING")

n_clipneg_style = N("CLIPTextEncode",[X3+550,Y3+480],[400,80],
    w=["text, letters, typography, font, label, caption, title, watermark, interface, UI, dashboard, app screen, window, dialog, popup, modal, menu, button, checkbox, list, grid, table, chart, form, toolbar, panel, frame, border, card, tooltip, palette swatch, color chart, infographic, poster, people, person, face, character, mascot, blurry, low quality, deformed, watermark"],
    title="Style Negative (hardened)",
    ins=[("clip","CLIP"),("text","STRING")],outs=[("CONDITIONING","CONDITIONING")])
C(n_lora,1,n_clipneg_style,0,"CLIP")

n_elat_style = N("EmptyLatentImage",[X3+550,Y3+580],[315,80],w=[480,832,1],
    title="Style Latent 16:9",outs=[("LATENT","LATENT")])

n_ks_style = N("KSampler",[X3+1000,Y3+380],[315,280],
    w=[123,"fixed",8,2.0,"dpmpp_sde","karras",1.0],
    title="KSampler Style (disabled)",
    ins=[("model","MODEL"),("positive","CONDITIONING"),("negative","CONDITIONING"),
         ("latent_image","LATENT")],
    outs=[("LATENT","LATENT")],
    mode=4)
C(n_lora,0,n_ks_style,0,"MODEL")
C(n_clippos_style,0,n_ks_style,1,"CONDITIONING")
C(n_clipneg_style,0,n_ks_style,2,"CONDITIONING")
C(n_elat_style,0,n_ks_style,3,"LATENT")

n_vd_style = N("VAEDecode",[X3+1350,Y3+380],[210,50],
    ins=[("samples","LATENT"),("vae","VAE")],outs=[("IMAGE","IMAGE")],
    mode=4)
C(n_ks_style,0,n_vd_style,0,"LATENT")
C(n_ckpt,2,n_vd_style,1,"VAE")

n_sv_style = N("SaveImage",[X3+1600,Y3+380],[315,80],w=["style_ref"],
    title="Save Style Ref (disabled)",
    ins=[("images","IMAGE"),("filename_prefix","STRING")],
    mode=4)
C(n_vd_style,0,n_sv_style,0,"IMAGE")

n_pv_style = N("PreviewImage",[X3+1600,Y3+480],[210,50],
    title="Preview Style (disabled)",ins=[("images","IMAGE")],
    mode=4)
C(n_vd_style,0,n_pv_style,0,"IMAGE")

# --- 3C: LLM generate enhanced negative prompt ---
n_negprompt = N("DirectLLMChat",[X3,Y3+680],[500,200],
    w=["Tao 1 dong negative prompt tieng Anh cho toan bo video.\nBat buoc phai bao gom cac nhom loi sau de giu mascot on dinh:\n- photorealistic human, realistic woman, realistic skin, realistic hair, human body\n- cropped head, cut off ears, cut off hands, cut off feet, off-center subject\n- extra limbs, bad anatomy, deformed face, melted outline, inconsistent mascot proportions\n- text, letters, typography, logo, watermark, UI, app screen, dashboard\n- cluttered background, noisy background, low contrast subject\nFormat: 1 dong, cac tu cach nhau boi dau phay.",
       "Tao negative prompt:","",16384,0.5],
    title="LLM: Negative Prompt",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_negprompt,0,"LLM_CONFIG")
C(n_concat2,0,n_negprompt,3,"STRING")

n_show_neg = N("easy showAnything",[X3+550,Y3+680],[250,80],w=["False"],
    title="Preview Negative",ins=[("anything","*",7)])
C(n_negprompt,0,n_show_neg,0,"STRING")

groups.append({"title":"STEP 3: REFERENCE SHEETS (Character + Style + Negative)","bounding":[X3-230,Y3-120,2100,1020],"color":"#a1309b"})

# =====================================================================
# STEP 4+5: MERGED SCENE PRODUCTION LOOP (Image Gen + Video Gen per scene)
#   - START image: txt2img with full prompt
#   - END image: img2img from START (denoise 0.65) for visual coherence
#   - Resize to WanVideo dimensions
#   - DIRECTLY feed resized images into WanVideo I2V (no LoadImage!)
#   - VACE Start-to-End frame control
#   - Direct decode output (no RIFE dependency)
# =====================================================================
X4, Y4 = 0, 1800

# --- Loop init ---
n_init4 = N("easy int",[X4,Y4],[210,58],w=[0],title="Counter",outs=[("INT","INT")])
n_first4 = N("easy int",[X4,Y4+70],[210,58],w=[1],title="First Iteration Threshold",outs=[("INT","INT")])

n_ls4 = N("easy whileLoopStart",[X4+260,Y4],[300,350],w=[True],
    title="Scene Production Loop",
    ins=[("condition","BOOLEAN"),
         ("initial_value0","*",7),  # counter
         ("initial_value1","*",7),  # start prompts
         ("initial_value2","*",7),  # end prompts
         ("initial_value3","*",7),  # motion prompts
         ("initial_value4","*",7)], # accumulated scene frames
    outs=[("flow","FLOW_CONTROL"),("value0","*"),("value1","*"),("value2","*"),("value3","*"),("value4","*")])
C(n_init4,0,n_ls4,1,"*")
C(n_prompt_start,0,n_ls4,2,"*")
C(n_prompt_end,0,n_ls4,3,"*")
C(n_prompt_motion,0,n_ls4,4,"*")
C(n_vd_char,0,n_ls4,5,"*")

# --- Get current START prompt ---
n_tl4_start = N("Text Load Line From File",[X4+620,Y4-80],[300,120],
    w=["","[filename]","TextBatch","index",0],
    title="Get Start Prompt",
    ins=[("multiline_text","STRING"),("file_path","STRING"),("dictionary_name","STRING"),
         ("mode","COMBO"),("index","INT")],
    outs=[("line_text","STRING"),("dictionary","DICT")])
C(n_ls4,2,n_tl4_start,0,"STRING")
C(n_ls4,1,n_tl4_start,4,"INT")

# --- Get current END prompt ---
n_tl4_end = N("Text Load Line From File",[X4+620,Y4+80],[300,120],
    w=["","[filename]","TextBatch","index",0],
    title="Get End Prompt",
    ins=[("multiline_text","STRING"),("file_path","STRING"),("dictionary_name","STRING"),
         ("mode","COMBO"),("index","INT")],
    outs=[("line_text","STRING"),("dictionary","DICT")])
C(n_ls4,3,n_tl4_end,0,"STRING")
C(n_ls4,1,n_tl4_end,4,"INT")

# --- Get current MOTION prompt ---
n_tl4_motion = N("Text Load Line From File",[X4+620,Y4+240],[300,120],
    w=["","[filename]","TextBatch","index",0],
    title="Get Motion Prompt",
    ins=[("multiline_text","STRING"),("file_path","STRING"),("dictionary_name","STRING"),
         ("mode","COMBO"),("index","INT")],
    outs=[("line_text","STRING"),("dictionary","DICT")])
C(n_ls4,4,n_tl4_motion,0,"STRING")
C(n_ls4,1,n_tl4_motion,4,"INT")

# --- CLIP Encode ---
n_clippos4s = N("CLIPTextEncode",[X4+980,Y4-140],[400,80],w=[""],
    title="Start Positive",
    ins=[("clip","CLIP"),("text","STRING")],outs=[("CONDITIONING","CONDITIONING")])
C(n_lora,1,n_clippos4s,0,"CLIP")
C(n_tl4_start,0,n_clippos4s,1,"STRING")

n_clipneg4 = N("CLIPTextEncode",[X4+980,Y4-50],[400,80],
    w=["photorealistic, realistic woman, realistic skin, realistic hair, human, live action, 3d render, bad anatomy, deformed, extra limbs, cropped head, cropped ears, cropped hands, cropped feet, off center subject, text, letters, typography, logo, watermark, interface, app screen, dashboard, cluttered background, noisy background"],
    title="Negative (from LLM)",
    ins=[("clip","CLIP"),("text","STRING")],outs=[("CONDITIONING","CONDITIONING")])
C(n_lora,1,n_clipneg4,0,"CLIP")
C(n_negprompt,0,n_clipneg4,1,"STRING")

n_clippos4e = N("CLIPTextEncode",[X4+980,Y4+740],[400,80],w=[""],
    title="End Positive",
    ins=[("clip","CLIP"),("text","STRING")],outs=[("CONDITIONING","CONDITIONING")])
C(n_lora,1,n_clippos4e,0,"CLIP")
C(n_tl4_end,0,n_clippos4e,1,"STRING")

# --- Generate START image (txt2img) ---
n_elat4s = N("EmptyLatentImage",[X4+980,Y4+40],[315,80],w=[480,832,1],
    title="Latent 16:9",outs=[("LATENT","LATENT")])

n_ks4s = N("KSampler",[X4+1430,Y4-140],[315,280],
    w=[42,"fixed",8,2.0,"dpmpp_sde","karras",1.0],
    title="KSampler START (Lightning: 8 steps, CFG 2.0)",
    ins=[("model","MODEL"),("positive","CONDITIONING"),("negative","CONDITIONING"),
         ("latent_image","LATENT")],
    outs=[("LATENT","LATENT")])
C(n_lora,0,n_ks4s,0,"MODEL")
C(n_clippos4s,0,n_ks4s,1,"CONDITIONING")
C(n_clipneg4,0,n_ks4s,2,"CONDITIONING")
C(n_elat4s,0,n_ks4s,3,"LATENT")

n_vd4s = N("VAEDecode",[X4+1800,Y4-140],[210,50],
    ins=[("samples","LATENT"),("vae","VAE")],outs=[("IMAGE","IMAGE")])
C(n_ks4s,0,n_vd4s,0,"LATENT")
C(n_ckpt,2,n_vd4s,1,"VAE")

# Resize START to WanVideo dimensions
n_rsz4s = N("WanVideoImageResizeToClosest",[X4+2050,Y4-140],[315,100],w=[480,832,"keep_input"],
    title="Resize START",
    ins=[("image","IMAGE")],
    outs=[("image","IMAGE"),("width","INT"),("height","INT")])
C(n_vd4s,0,n_rsz4s,0,"IMAGE")

n_sv4s = N("SaveImage",[X4+2400,Y4-140],[315,80],w=["scene_start"],
    title="Save START",ins=[("images","IMAGE"),("filename_prefix","STRING")])
C(n_rsz4s,0,n_sv4s,0,"IMAGE")

n_pv4s = N("PreviewImage",[X4+2400,Y4-50],[210,50],
    title="Preview START",ins=[("images","IMAGE")])
C(n_rsz4s,0,n_pv4s,0,"IMAGE")

# --- Generate END image (img2img from START for coherence) ---
n_ve4e = N("VAEEncode",[X4+1800,Y4+600],[210,50],
    title="Encode START -> Latent",
    ins=[("pixels","IMAGE"),("vae","VAE")],outs=[("LATENT","LATENT")])
C(n_vd4s,0,n_ve4e,0,"IMAGE")
C(n_ckpt,2,n_ve4e,1,"VAE")

n_ks4e = N("KSampler",[X4+1430,Y4+600],[315,280],
    w=[42,"fixed",10,2.0,"dpmpp_sde","karras",0.55],
    title="KSampler END (Lightning: 10 steps, CFG 2.0, denoise 0.55)",
    ins=[("model","MODEL"),("positive","CONDITIONING"),("negative","CONDITIONING"),
         ("latent_image","LATENT")],
    outs=[("LATENT","LATENT")])
C(n_lora,0,n_ks4e,0,"MODEL")
C(n_clippos4e,0,n_ks4e,1,"CONDITIONING")
C(n_clipneg4,0,n_ks4e,2,"CONDITIONING")
C(n_ve4e,0,n_ks4e,3,"LATENT")

n_vd4e = N("VAEDecode",[X4+2050,Y4+600],[210,50],
    ins=[("samples","LATENT"),("vae","VAE")],outs=[("IMAGE","IMAGE")])
C(n_ks4e,0,n_vd4e,0,"LATENT")
C(n_ckpt,2,n_vd4e,1,"VAE")

n_rsz4e = N("WanVideoImageResizeToClosest",[X4+2050,Y4+680],[315,100],w=[480,832,"keep_input"],
    title="Resize END",
    ins=[("image","IMAGE")],
    outs=[("image","IMAGE"),("width","INT"),("height","INT")])
C(n_vd4e,0,n_rsz4e,0,"IMAGE")

n_sv4e = N("SaveImage",[X4+2400,Y4+600],[315,80],w=["scene_end"],
    title="Save END",ins=[("images","IMAGE"),("filename_prefix","STRING")])
C(n_rsz4e,0,n_sv4e,0,"IMAGE")

n_pv4e = N("PreviewImage",[X4+2400,Y4+700],[210,50],
    title="Preview END",ins=[("images","IMAGE")])
C(n_rsz4e,0,n_pv4e,0,"IMAGE")

# --- VACE: Start-to-End frame transition (DIRECT from resize, no LoadImage!) ---
n_vace = N("WanVideoVACEStartToEndFrame",[X4+2750,Y4-140],[380,200],
    w=[49,0.5],
    title="VACE Start->End Control",
    ins=[("num_frames","INT"),("empty_frame_level","FLOAT"),
         ("start_image","IMAGE",7),("end_image","IMAGE",7),
         ("control_images","IMAGE",7),("inpaint_mask","MASK",7),
         ("start_index","INT",7),("end_index","INT",7)],
    outs=[("images","IMAGE"),("masks","MASK")])
C(n_nf,0,n_vace,0,"INT")
C(n_rsz4s,0,n_vace,2,"IMAGE")
C(n_rsz4e,0,n_vace,3,"IMAGE")

# --- Combine scene + motion prompt ---
n_concat5 = N("Text Concatenate",[X4+2750,Y4+100],[300,100],
    w=[". ","true"],
    title="Scene + Motion Prompt",
    ins=[("delimiter","STRING"),("clean_whitespace","COMBO"),
         ("text_a","STRING",7),("text_b","STRING",7)],
    outs=[("STRING","STRING")])
C(n_tl4_start,0,n_concat5,2,"STRING")
C(n_tl4_motion,0,n_concat5,3,"STRING")

# --- CLIP Vision Encode ---
n_cve5 = N("WanVideoClipVisionEncode",[X4+2750,Y4+240],[350,200],
    w=[1.0,1.0,"center","average",True],
    title="CLIP Vision Encode",
    ins=[("clip_vision","CLIP_VISION"),("image_1","IMAGE"),
         ("image_2","IMAGE",7),("negative_image","IMAGE",7)],
    outs=[("image_embeds","WANVIDIMAGE_CLIPEMBEDS")])
C(n_clipv,0,n_cve5,0,"CLIP_VISION")
C(n_rsz4s,0,n_cve5,1,"IMAGE")
C(n_rsz4e,0,n_cve5,2,"IMAGE")

# --- I2V Encode + VACE ---
n_i2v5 = N("WanVideoImageToVideoEncode",[X4+3150,Y4+240],[380,220],
    w=[480,832,49,0.0,1.0,1.0,True],
    title="I2V Encode + VACE",
    ins=[("vae","WANVAE",7),("clip_embeds","WANVIDIMAGE_CLIPEMBEDS",7),
         ("start_image","IMAGE",7),("end_image","IMAGE",7),
         ("control_embeds","WANVIDIMAGE_EMBEDS",7),
         ("temporal_mask","MASK",7),("extra_latents","LATENT",7)],
    outs=[("image_embeds","WANVIDIMAGE_EMBEDS")])
C(n_wanvae,0,n_i2v5,0,"WANVAE")
C(n_cve5,0,n_i2v5,1,"WANVIDIMAGE_CLIPEMBEDS")
C(n_rsz4s,0,n_i2v5,2,"IMAGE")
C(n_rsz4e,0,n_i2v5,3,"IMAGE")
C(n_vace,1,n_i2v5,5,"MASK")

# --- WanVideo Text Encode ---
n_te5 = N("WanVideoTextEncodeCached",[X4+3150,Y4-140],[400,200],
    w=["umt5_xxl_fp16.safetensors","bf16",
       "",
       "blurry, static, low quality, distorted, jerky motion, flickering, frozen, duplicate frames, morphing, face distortion, unnatural movement, abrupt cuts, inconsistent lighting, floating objects",
       "disabled",True,"gpu"],
    title="WanVideo Text Encode",
    ins=[("positive_prompt","STRING"),
         ("extender_args","WANVIDEOPROMPTEXTENDER_ARGS",7)],
    outs=[("text_embeds","WANVIDEOTEXTEMBEDS"),("negative_text_embeds","WANVIDEOTEXTEMBEDS"),
          ("positive_prompt","STRING")])
C(n_concat5,0,n_te5,0,"STRING")

# --- WanVideo Sampler ---
n_ws5 = N("WanVideoSampler",[X4+3600,Y4-140],[380,350],
    w=[30,6.0,8.0,42,"randomize",True,"dpm++_sde",0],
    title="WanVideo 2.2 Sampler (30 steps, v3)",
    ins=[("model","WANVIDEOMODEL"),("image_embeds","WANVIDIMAGE_EMBEDS"),
         ("text_embeds","WANVIDEOTEXTEMBEDS",7),("samples","LATENT",7)],
    outs=[("samples","LATENT"),("denoised_samples","LATENT")])
C(n_wanmod,0,n_ws5,0,"WANVIDEOMODEL")
C(n_i2v5,0,n_ws5,1,"WANVIDIMAGE_EMBEDS")
C(n_te5,0,n_ws5,2,"WANVIDEOTEXTEMBEDS")

# --- Decode ---
n_dec5 = N("WanVideoDecode",[X4+4030,Y4-140],[315,200],
    w=[False,272,272,144,128],
    title="WanVideo Decode",
    ins=[("vae","WANVAE"),("samples","LATENT")],outs=[("images","IMAGE")])
C(n_wanvae,0,n_dec5,0,"WANVAE")
C(n_ws5,0,n_dec5,1,"LATENT")

# --- Save scene video ---
n_vc5 = N("VHS_VideoCombine",[X4+4400,Y4-140],[400,250],
    w=[24.0,0,"scene_video","video/h264-mp4",False,True],
    title="Save Scene Video",
    ins=[("images","IMAGE"),("audio","AUDIO",7)],
    outs=[("Filenames","VHS_FILENAMES")])
C(n_dec5,0,n_vc5,0,"IMAGE")

# --- v3: Last Frame Extractor (Sora2Alike scene chaining) ---
# Extracts final frame so it can be wired as start_image of next scene for continuity.
# KJNodes GetImageRangeFromBatch supports start_index=-1 (from end).
n_getimg_last = N("GetImageRangeFromBatch",[X4+4400,Y4+320],[300,100],
    w=[-1,1],
    title="Last Frame from Video",
    ins=[("images","IMAGE"),("start_index","INT"),("num_frames","INT")],
    outs=[("IMAGE","IMAGE"),("MASK","MASK")])
C(n_dec5,0,n_getimg_last,0,"IMAGE")

n_sv_last = N("SaveImage",[X4+4750,Y4+320],[300,80],
    w=["scene_last_frame"],
    title="Save Last Frame (for chaining)",
    ins=[("images","IMAGE"),("filename_prefix","STRING")])
C(n_getimg_last,0,n_sv_last,0,"IMAGE")

n_pv5 = N("PreviewImage",[X4+4400,Y4+160],[210,50],
    title="Preview Video Frame",ins=[("images","IMAGE")])
C(n_dec5,0,n_pv5,0,"IMAGE")

# --- Loop control ---
n_math4 = N("MathExpression|pysssss",[X4+4400,Y4+300],[250,80],w=["a + 1"],
    title="Counter + 1",
    ins=[("a","*",7),("b","*",7),("c","*",7)],outs=[("INT","INT"),("FLOAT","FLOAT")])
C(n_ls4,1,n_math4,0,"*")

n_cmp4 = N("easy compare",[X4+4400,Y4+410],[250,100],w=["a < b"],
    ins=[("a","*",7),("b","*",7),("comparison","COMBO")],outs=[("boolean","BOOLEAN")])
C(n_math4,0,n_cmp4,0,"*")
C(n_scene_count,0,n_cmp4,1,"*")

n_first_scene4 = N("easy compare",[X4+4400,Y4+530],[250,100],w=["a < b"],
    title="Is First Scene?",
    ins=[("a","*",7),("b","*",7),("comparison","COMBO")],outs=[("boolean","BOOLEAN")])
C(n_ls4,1,n_first_scene4,0,"*")
C(n_first4,0,n_first_scene4,1,"*")

n_batch_accum4 = N("ImageBatch",[X4+4400,Y4+660],[210,50],
    title="Append Scene Frames",
    ins=[("image1","IMAGE"),("image2","IMAGE")],
    outs=[("IMAGE","IMAGE")])
C(n_ls4,5,n_batch_accum4,0,"IMAGE")
C(n_dec5,0,n_batch_accum4,1,"IMAGE")

n_pick_accum4 = N("easy ifElse",[X4+4700,Y4+640],[250,100],
    title="Pick Accumulated Frames",
    ins=[("on_true","*"),("on_false","*"),("boolean","BOOLEAN")],
    outs=[("*","*")])
C(n_dec5,0,n_pick_accum4,0,"*")
C(n_batch_accum4,0,n_pick_accum4,1,"*")
C(n_first_scene4,0,n_pick_accum4,2,"BOOLEAN")

n_le4 = N("easy whileLoopEnd",[X4+4700,Y4+250],[300,350],w=[False],
    title="Scene Production Loop End",
    ins=[("flow","FLOW_CONTROL"),("condition","BOOLEAN"),
         ("initial_value0","*",7),("initial_value1","*",7),
         ("initial_value2","*",7),("initial_value3","*",7),
         ("initial_value4","*",7)],
    outs=[("value0","*"),("value1","*"),("value2","*"),("value3","*"),("value4","*")])
C(n_ls4,0,n_le4,0,"FLOW_CONTROL")
C(n_cmp4,0,n_le4,1,"BOOLEAN")
C(n_math4,0,n_le4,2,"*")     # counter
C(n_ls4,2,n_le4,3,"*")       # start prompts
C(n_ls4,3,n_le4,4,"*")       # end prompts
C(n_ls4,4,n_le4,5,"*")       # motion prompts (was missing!)
C(n_pick_accum4,0,n_le4,6,"*")

groups.append({"title":"STEP 4+5: SCENE PRODUCTION (Image + Video per Scene)","bounding":[X4-30,Y4-380,5100,1200],"color":"#3f789e"})

# =====================================================================
# STEP 6: VOICE GENERATION + SRT TEXT
# =====================================================================
X6, Y6 = 0, 3200

n_chat6 = N("DirectLLMChat",[X6+200,Y6],[550,300],
    w=["Tu kich ban video, trich xuat voiceover cua tat ca cac canh.\n\nYeu cau:\n- Ghep thanh mot doan van LIEN MACH, doc duoc tu nhien nhu narrator chuyen nghiep\n- Giu nguyen ngon ngu goc\n- Them ngat nghi tu nhien giua cac canh (dung dau ... hoac dau ,)\n- Dieu chinh nhip doc: canh hook/climax nhanh hon, canh calm cham hon\n- CHI TRA VE VAN BAN THUAN TUY, khong format, khong danh so\n\nChu y: Van ban nay se duoc doc boi AI voice (ElevenLabs/ChatterBox),\nnen viet sao cho doc len tu nhien, tranh cau dai qua 30 tu.",
       "Trich xuat voiceover:","",16384,0.7],
    title="LLM: Extract Voiceover",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_chat6,0,"LLM_CONFIG")
C(n_script,0,n_chat6,3,"STRING")

n_show6 = N("easy showAnything",[X6+800,Y6],[280,80],w=["False"],
    title="Preview Voiceover Text",ins=[("anything","*",7)])
C(n_chat6,0,n_show6,0,"STRING")

# ElevenLabs Voice Selector → TTS chain
n_voice_sel = N("ElevenLabsVoiceSelector",[X6+200,Y6+350],[350,80],
    w=["Daniel (male, british)"],
    title="ElevenLabs Voice",
    outs=[("voice","ELEVENLABS_VOICE")])

n_eleven = N("ElevenLabsTextToSpeech",[X6+200,Y6+460],[450,250],
    w=[0.5,"auto","eleven_v3",1.0,0.75,"",42,"mp3_44100_192"],
    title="ElevenLabs TTS",
    ins=[("voice","ELEVENLABS_VOICE"),("text","STRING")],
    outs=[("AUDIO","AUDIO")])
C(n_voice_sel,0,n_eleven,0,"ELEVENLABS_VOICE")
C(n_chat6,0,n_eleven,1,"STRING")

n_voicenote = N("Text Multiline",[X6+200,Y6+640],[450,100],
    w=["ELEVENLABS VOICES GOI Y:\n- Daniel (JBFqnCBsd6RMkjVDRZzb): narrator tram, cinematic\n- Rachel (21m00Tcm4TlvDq8ikWAM): nu, am ap, storytelling\n- Antoni (ErXwobaYiN019PkySvjV): nam, tre, energetic\n- Bella (EXAVITQu4vr4xnSDxMaL): nu, diu dang, intimate"],
    title="Voice Guide",color="#432",bgcolor="#653",
    outs=[("STRING","STRING")])

n_refaudio = N("LoadAudio",[X6+700,Y6+640],[315,80],
    w=["reference_voice.mp3"],title="Ref Voice (for Cloning)",
    outs=[("AUDIO","AUDIO")],mode=4)

n_audiodur = N("AudioDurationNode",[X6+1250,Y6+350],[300,100],
    w=[24.0,"both"],title="Audio Duration",
    ins=[("audio","AUDIO"),("fps","FLOAT"),("output_format","COMBO")],
    outs=[("duration_seconds","FLOAT"),("frame_count","INT"),("info","STRING")])
C(n_eleven,0,n_audiodur,0,"AUDIO")

n_show6dur = N("easy showAnything",[X6+1250,Y6+470],[250,80],w=["False"],
    title="Duration Info",ins=[("anything","*",7)])
C(n_audiodur,0,n_show6dur,0,"STRING")

n_vcaudio6 = N("VHS_VideoCombine",[X6+1250,Y6+580],[400,150],
    w=[24.0,0,"voiceover_output","audio/flac",False,True],
    title="Save Voiceover",
    ins=[("images","IMAGE",7),("audio","AUDIO",7)],
    outs=[("Filenames","VHS_FILENAMES")])
C(n_eleven,0,n_vcaudio6,1,"AUDIO")

n_srt = N("DirectLLMChat",[X6+1250,Y6],[550,300],
    w=["Tu van ban voiceover, tao file SRT subtitle.\nUoc luong thoi gian doc moi cau dua tren so tu (trung binh 150 tu/phut tieng Viet, 160 tu/phut tieng Anh).\n\nFormat SRT CHINH XAC:\n1\n00:00:00,000 --> 00:00:03,500\nCau dau tien cua voiceover\n\n2\n00:00:03,800 --> 00:00:07,200\nCau tiep theo\n\nQuy tac:\n- Moi subtitle toi da 2 dong, moi dong toi da 42 ky tu\n- Thoi gian hien thi toi thieu 1.5 giay, toi da 5 giay\n- Khoang cach giua 2 subtitle toi thieu 0.2 giay\n- Chia tai dau cau hoac dau phay, KHONG chia giua tu\n- Tong thoi gian phu hop voi video 60 giay",
       "Tao SRT subtitle:","",16384,0.5],
    title="LLM: Generate SRT",
    ins=[("config","LLM_CONFIG"),("system_prompt","STRING"),("prompt","STRING"),
         ("context_text","STRING",7),("max_tokens","INT"),("temperature","FLOAT")],
    outs=[("text","STRING")])
C(n_llm,0,n_srt,0,"LLM_CONFIG")
C(n_chat6,0,n_srt,3,"STRING")

n_show_srt = N("easy showAnything",[X6+1850,Y6],[250,80],w=["False"],
    title="Preview SRT",ins=[("anything","*",7)])
C(n_srt,0,n_show_srt,0,"STRING")

groups.append({"title":"STEP 6: VOICE (ElevenLabs) + SRT Text","bounding":[X6-80,Y6-60,2200,860],"color":"#b06634"})

# =====================================================================
# STEP 7: DYNAMIC ASSEMBLY & FINAL OUTPUT
# =====================================================================
X7, Y7 = 0, 4400
NUM_SCENES = 11  # Default matches the current 11-scene input script

# --- Use accumulated loop frames directly ---
n_pv_batch = N("PreviewImage",[X7,Y7],[210,50],
    title="Preview Accumulated Frames",ins=[("images","IMAGE")])
C(n_le4,4,n_pv_batch,0,"IMAGE")

# --- Final video with voice ---
n_vcfinal = N("VHS_VideoCombine",[X7+470, Y7+100],[400,300],
    w=[24.0, 0, "eStudio_final_with_voice", "video/h264-mp4", False, True],
    title="FINAL VIDEO (Voice)",
    ins=[("images","IMAGE"),("audio","AUDIO",7)],
    outs=[("Filenames","VHS_FILENAMES")])
C(n_le4, 4, n_vcfinal, 0, "IMAGE")
C(n_eleven, 0, n_vcfinal, 1, "AUDIO")

n_pvfinal = N("PreviewImage",[X7+470, Y7+440],[210,50],
    title="Preview Final",ins=[("images","IMAGE")])
C(n_le4, 4, n_pvfinal, 0, "IMAGE")

# --- Raw video NO subtitles (for CapCut post-production) ---
n_vcraw = N("VHS_VideoCombine",[X7+470, Y7+540],[400,250],
    w=[24.0, 0, "eStudio_raw_for_capcut", "video/h264-mp4", False, True],
    title="RAW VIDEO (for CapCut)",
    ins=[("images","IMAGE"),("audio","AUDIO",7)],
    outs=[("Filenames","VHS_FILENAMES")])
C(n_le4, 4, n_vcraw, 0, "IMAGE")
C(n_eleven, 0, n_vcraw, 1, "AUDIO")

n_capcut_note = N("Text Multiline",[X7+470, Y7+850],[450,180],
    w=["CAPCUT POST-PRODUCTION GUIDE:\n1. Import: eStudio_raw_for_capcut.mp4 + voiceover_output.flac\n2. Import SRT: Copy noi dung SRT tu Preview vao file .srt\n3. Music: Them nhac nen tu CapCut library (lofi, cinematic, ambient)\n4. SFX: Them sound effects theo tung canh (footsteps, rain, city)\n5. Pacing: Dieu chinh toc do tung clip (speed ramping)\n6. Transitions: Them hieu ung chuyen canh (dissolve, whip pan)\n7. Color grade: LUT cinematic (Teal & Orange, Warm Film)\n8. Export: 1080p H.264, bitrate 10-15 Mbps"],
    title="CapCut Guide",color="#432",bgcolor="#653",
    outs=[("STRING","STRING")])

groups.append({"title":f"STEP 7: POST-PRODUCTION & FINAL ASSEMBLY ({NUM_SCENES} scenes)","bounding":[X7-30,Y7-60,1500,1120],"color":"#4a4"})

# =====================================================================
# OUTPUT JSON
# =====================================================================
print(json.dumps({
    "id":str(uuid.uuid4()),"revision":0,
    "last_node_id":_nid,"last_link_id":_lid,
    "nodes":nodes,"links":links,"groups":groups,
    "config":{},"extra":{"ds":{"scale":0.25,"offset":[0,0]}},
    "version":0.4
},indent=2,ensure_ascii=False))
