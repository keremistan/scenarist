from collections.abc import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import pdfplumber
import regex as re

class ScreenplayLoader(BaseLoader):
    
    def __init__(self, file_path) -> None:
        super().__init__()
        
        self.file_path = file_path
        self.scene_regex = "(?:INT\\.|EXT\\.)\\s([A-Z0-9\\s\\-\\']+)"

        
    def lazy_load(self) -> Iterator[Document]:
        whole_doc = ""

        with pdfplumber.open(self.file_path) as fp:
            for i, page in enumerate(fp.pages):
                text_in_page = page.extract_text()
                whole_doc += text_in_page
        
        
        matches = list(re.finditer(self.scene_regex, whole_doc))
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(whole_doc)
            print("current index {}; start: {}; end: {}; len(matches): {}; len(whole_doc): {}".format(
                i, start, end, len(matches), len(whole_doc)
            ))

            current_scene = whole_doc[start:end]
            scene_header = match.group().strip()
        
            yield Document(
                page_content=current_scene,
                metadata={
                    "file_name": self.file_path,
                    "scene_index": i+1,
                    "scene_header": scene_header,
                    "type": "scene"
                }
            )


