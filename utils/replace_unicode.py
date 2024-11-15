import re
import unicodedata

def replace_unicode(text, replace_with=" "):
    # This regex identifies Unicode characters outside the ASCII range.
    unicode_chars = re.compile(r'[^\x00-\x7F]+')
    
    def to_ascii(match):
        char = match.group(0)
        try:
            # Attempt to decompose Unicode character into ASCII equivalent
            ascii_equiv = unicodedata.normalize('NFKD', char).encode('ascii', 'ignore').decode('ascii')
            return ascii_equiv if ascii_equiv else replace_with
        except UnicodeEncodeError:
            # Replace with specified placeholder if conversion fails
            return replace_with
    
    # Replace each Unicode character with either an ASCII equivalent or a placeholder
    return unicode_chars.sub(to_ascii, text)