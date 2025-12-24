from collections.abc import Iterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import pdfplumber
import regex as re

class ScreenplayLoader(BaseLoader):
    
    def __init__(self, file_path) -> None:
        super().__init__()
        
        self.file_path = file_path
        # self.scene_regex = "(?:INT\\.|EXT\\.)\\s([A-Z0-9]+)(?:\\s-\\s[A-Z]+)" #TODO: to be checked for its validity
        self.scene_regex = "(?:INT\\.|EXT\\.)\\s([A-Z0-9\\s\\-]+)" #TODO: to be checked for its validity

        
    def lazy_load(self) -> Iterator[Document]:
        whole_doc = ""

        with pdfplumber.open(self.file_path) as fp:
            # fp.pages[0].extract_text()
            for i, page in enumerate(fp.pages):
                # print("starting to process page {}".format(i))
                text_in_page = page.extract_text()
                # print(text_in_page)
                
                whole_doc += text_in_page
                # print("appended\n")
        
        
        matches = list(re.finditer(self.scene_regex, whole_doc))
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start()
            
            print("start index: {}".format(start))
            print("end index: {}".format(end))
            
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


