import base64

def encode_base64_contentl(content_url: str) -> str:
    """Encode a content retrieved from a local url to base64 format"""

    with open(content_url, 'rb') as f:
        result = base64.b64encode(f.read()).decode('utf-8')
    return result
