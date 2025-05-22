from argparse import _SubParsersAction
import time
from console import Console
from rich.progress import Progress
from mode import Mode
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import re
from langchain.schema import Document


class LoadFindTheMomentMode(Mode):
    def __init__(
        self, 
        console: Console, 
        youtube_link: str, 
        verbose: bool = False):
        super().__init__(console)

        self.youtube_link = youtube_link
        self.verbose = verbose

    @staticmethod
    def add_subparser(name: str, subparser: _SubParsersAction):
        load_book_subparser = subparser.add_parser(name)
        load_book_subparser.add_argument("youtube_link", type=str, help="The youtube link to load")
        load_book_subparser.add_argument("--verbose", "-v", action="store_true", help="Verbose mode")


    def run(self):
        embedding = OllamaEmbeddings(model="mxbai-embed-large:latest")

        # Create vector store
        vector_store = Chroma(
            embedding_function=embedding,
            persist_directory="./store"
        )

        # Loading
        text_splitter = SemanticChunker(
            embeddings=embedding,
        )

        video_id = self.extract_video_id(self.youtube_link)

        ytt_api = YouTubeTranscriptApi()
        
        try:
            fetched_transcript = ytt_api.fetch(video_id, languages=['fr', 'en'])
        except Exception as e:
            self.console.print(f"❌ Impossible de récupérer la transcription : {e}")
            return
        
        sentences = []

        for i in range(len(fetched_transcript)):
            splite = re.split(r'[.!?]', fetched_transcript[i].text)
            for j in range(len(splite)):
                if sentences == []:
                    sentences.append({
                        "time": self.format_seconds(fetched_transcript[i].start),
                        "text": splite[j]
                    })
                else:
                    if j == 0:
                        sentences[-1]["text"] += " " + splite[j]
                    else:
                        sentences.append({
                        "time": self.format_seconds(fetched_transcript[i].start),
                        "text": splite[j]
                    })
        
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Chargement de la transcription...", total=len(sentences))
            # print(sentences)
            for sentence in sentences:
                doc = Document(page_content=sentence["text"], metadata={
                    "time": sentence["time"],
                    "video_id": self.extract_video_id(self.youtube_link),
                })
                chunks = text_splitter.split_documents([doc])
                vector_store.add_documents(chunks)
                progress.update(task, advance=1)

        self.console.print("[green]✅ Transcription chargée avec succès.")
                

    def extract_video_id(self, url):
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get('v', [None])[0]

    def format_seconds(self, seconds):
        heures = int(seconds) // 3600
        minutes = (int(seconds) % 3600) // 60
        remaining_seconds = int(seconds) % 60

        if heures > 0:
            return f"{heures}:{minutes:02d}:{remaining_seconds:02d}"
        else:
            return f"{minutes}:{remaining_seconds:02d}"

    def firstCarIsupper(self, chaine):
        if chaine == "":
            return False
        return chaine[0].strip().isupper()
        
        
