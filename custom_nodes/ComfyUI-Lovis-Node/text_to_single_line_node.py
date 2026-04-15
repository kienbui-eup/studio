class TextToSingleLineNode:
    """
    A ComfyUI node that converts multi-line text to single line text by replacing
    line breaks with periods and spaces.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "separator": ("STRING", {
                    "default": ". ",
                    "multiline": False
                }),
                "trim_lines": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Trim Whitespace",
                    "label_off": "Keep Whitespace"
                }),
                "skip_empty_lines": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Skip Empty",
                    "label_off": "Keep Empty"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("flattened_text",)
    FUNCTION = "flatten_text"
    CATEGORY = "text/utility"
    
    def flatten_text(self, text, separator, trim_lines, skip_empty_lines):
        """
        Convert multi-line text to single line text.
        
        Args:
            text (str): The input multi-line text
            separator (str): The separator to use between lines (default: ". ")
            trim_lines (bool): Whether to trim whitespace from each line
            skip_empty_lines (bool): Whether to skip empty lines
            
        Returns:
            tuple: (flattened_text,)
        """
        if not text:
            return ("",)
        
        # Split text into lines
        lines = text.split('\n')
        
        # Process each line
        processed_lines = []
        for line in lines:
            if trim_lines:
                line = line.strip()
            
            # Skip empty lines if requested
            if skip_empty_lines and not line:
                continue
                
            processed_lines.append(line)
        
        # Join lines with the separator
        flattened = separator.join(processed_lines)
        
        # Clean up any double separators that might occur
        while separator + separator in flattened:
            flattened = flattened.replace(separator + separator, separator)
        
        return (flattened,)
