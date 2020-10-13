import tika
from tika import parser

def extract(file_path):
    headers = {
        'X-Tika-PDFextractInlineImages': 'true',
        "X-Tika-OCRLanguage": "eng+swe"
    }
    parsed = parser.from_file(file_path, serverEndpoint="http://localhost:9898/tika", headers=headers)
    return parsed

scanned_doc = "assets/sample_scanned/PublicWaterMassMailing.pdf"
path = "/mnt/InternalStorage/sidkas/skr/downloaded_pdfs/Ljusdals kommun-Protokoll%20UN%202017-04-06.pdf"
data = extract(path)

lines = data.get("content").split('\n')
print(lines[100:140])