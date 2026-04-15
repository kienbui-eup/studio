class LineCountNode:
    """
    A ComfyUI node that counts the number of lines in a text input.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "include_empty_lines": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Include Empty",
                    "label_off": "Skip Empty"
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("line_count", "summary")
    FUNCTION = "count_lines"
    CATEGORY = "text/utility"
    
    def count_lines(self, text, include_empty_lines):
        """
        Count the number of lines in the input text.
        
        Args:
            text (str): The input text to count lines for
            include_empty_lines (bool): Whether to include empty lines in the count
            
        Returns:
            tuple: (line_count, summary_string)
        """
        if not text:
            return (0, "No text provided - 0 lines")
        
        # Split text into lines
        lines = text.split('\n')
        
        if include_empty_lines:
            line_count = len(lines)
            summary = f"Total lines: {line_count} (including empty lines)"
        else:
            # Filter out empty lines (lines that are empty or contain only whitespace)
            non_empty_lines = [line for line in lines if line.strip()]
            line_count = len(non_empty_lines)
            total_lines = len(lines)
            empty_lines = total_lines - line_count
            summary = f"Non-empty lines: {line_count} (total: {total_lines}, empty: {empty_lines})"
        
        return (line_count, summary)
