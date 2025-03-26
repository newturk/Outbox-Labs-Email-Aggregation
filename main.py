import os
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from imap_tools import MailBox, AND
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
import openai
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import requests
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
class Config:
    def __init__(self):
        self.es = Elasticsearch("http://localhost:9200")
        self.openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.slack = WebClient(token=os.getenv("SLACK_TOKEN"))
        self.webhook_url = os.getenv("WEBHOOK_URL")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma = chromadb.Client()
        self.email_collection = self.chroma.get_or_create_collection("emails")
        self.knowledge_collection = self.chroma.get_or_create_collection("outreach_knowledge")

config = Config()

# Data Models
class EmailAccount(BaseModel):
    email: str
    password: str
    imap_server: str
    imap_port: int = 993

class Email(BaseModel):
    uid: str
    subject: str
    body: str
    from_: str
    to: str
    date: datetime
    folder: str = "INBOX"
    account: str
    category: Optional[str] = None

class SearchQuery(BaseModel):
    text: str
    account: Optional[str] = None
    folder: Optional[str] = None
    category: Optional[str] = None

# Services
class IMAPService:
    def __init__(self):
        self.active_connections = {}

    async def connect_account(self, account: EmailAccount):
        mailbox = MailBox(account.imap_server)
        mailbox.login(account.email, account.password, initial_folder="INBOX")
        self.active_connections[account.email] = mailbox
        return mailbox

    async def fetch_emails(self, account: EmailAccount, days: int = 30):
        mailbox = await self.connect_account(account)
        since_date = datetime.now() - timedelta(days=days)
        emails = []
        
        for msg in mailbox.fetch(AND(date_gte=since_date)):
            email = Email(
                uid=msg.uid,
                subject=msg.subject,
                body=msg.text or msg.html,
                from_=msg.from_,
                to=msg.to,
                date=msg.date,
                account=account.email
            )
            emails.append(email)
        
        return emails

    async def idle(self, account: EmailAccount, callback):
        """Monitor mailbox for new emails in real-time"""
        mailbox = await self.connect_account(account)
        
        while True:
            responses = mailbox.idle.wait(timeout=300)
            if responses:
                new_emails = await self.fetch_emails(account, days=1)
                for email in new_emails:
                    callback(email)
            
            await asyncio.sleep(1)

class EmailSearchService:
    def __init__(self):
        self.es = config.es
        self.create_index()

    def create_index(self):
        if not self.es.indices.exists(index="emails"):
            self.es.indices.create(
                index="emails",
                body={
                    "mappings": {
                        "properties": {
                            "subject": {"type": "text"},
                            "body": {"type": "text"},
                            "from": {"type": "keyword"},
                            "to": {"type": "keyword"},
                            "date": {"type": "date"},
                            "folder": {"type": "keyword"},
                            "account": {"type": "keyword"},
                            "category": {"type": "keyword"}
                        }
                    }
                }
            )

    def index_email(self, email: Email):
        doc = email.dict()
        doc["from"] = doc.pop("from_")
        self.es.index(index="emails", document=doc)

    def search(self, query: SearchQuery):
        must = []
        if query.text:
            must.append({"match": {"body": query.text}})
        
        filters = []
        if query.account:
            filters.append({"term": {"account": query.account}})
        if query.folder:
            filters.append({"term": {"folder": query.folder}})
        if query.category:
            filters.append({"term": {"category": query.category}})
        
        body = {
            "query": {
                "bool": {
                    "must": must,
                    "filter": filters
                }
            }
        }
        
        result = self.es.search(index="emails", body=body)
        return [Email(**hit["_source"]) for hit in result["hits"]["hits"]]

class AICategorizer:
    def categorize(self, email: Email) -> str:
        prompt = f"""Categorize this email into one of these categories:
        - Interested
        - Meeting Booked
        - Not Interested
        - Spam
        - Out of Office
        
        Email:
        Subject: {email.subject}
        Body: {email.body[:1000]}
        
        Respond only with the category name."""
        
        response = config.openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        category = response.choices[0].message.content
        return category

class NotificationService:
    def send_slack_notification(self, email: Email):
        try:
            config.slack.chat_postMessage(
                channel="#email-notifications",
                text=f"New Interested Email from {email.from_}",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*New Interested Email*\n*From:* {email.from_}\n*Subject:* {email.subject}"
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Preview:* {email.body[:200]}..."
                        }
                    }
                ]
            )
        except SlackApiError as e:
            print(f"Slack error: {e}")

    def trigger_webhook(self, email: Email):
        try:
            requests.post(
                config.webhook_url,
                json={
                    "event": "email_interested",
                    "data": email.dict()
                },
                timeout=5
            )
        except Exception as e:
            print(f"Webhook error: {e}")

class ReplySuggestor:
    def __init__(self):
        self.embedder = config.embedder
        self.collection = config.knowledge_collection

    def add_knowledge(self, text: str):
        embedding = self.embedder.encode(text).tolist()
        self.collection.add(
            ids=[str(len(self.collection.get()["ids"]) + 1],
            documents=[text],
            embeddings=[embedding]
        )

    def suggest_reply(self, email: Email) -> str:
        # Get relevant context
        query_embedding = self.embedder.encode(email.body).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        
        context = results["documents"][0][0]
        
        # Generate reply
        prompt = f"""Generate a professional reply to this email using the provided context.
        
        Email:
        Subject: {email.subject}
        Body: {email.body[:1000]}
        
        Context:
        {context}
        
        Suggested Reply:"""
        
        response = config.openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        return response.choices[0].message.content

# Initialize services
imap_service = IMAPService()
search_service = EmailSearchService()
categorizer = AICategorizer()
notifier = NotificationService()
reply_suggestor = ReplySuggestor()

# Store some initial knowledge
reply_suggestor.add_knowledge(
    "Our product helps with cold outreach automation. "
    "For interested leads, share the booking link: https://cal.com/example"
)

# API Endpoints
@app.post("/accounts/connect")
async def connect_account(account: EmailAccount, background_tasks: BackgroundTasks):
    """Connect to an IMAP account and start syncing emails"""
    background_tasks.add_task(imap_service.idle, account, process_new_email)
    return {"status": "connected"}

@app.get("/emails/search")
async def search_emails(query: SearchQuery):
    """Search emails with filters"""
    return search_service.search(query)

@app.post("/emails/suggest-reply")
async def suggest_reply(email: Email):
    """Get AI-generated reply suggestion"""
    return {"reply": reply_suggestor.suggest_reply(email)}

# Helper Functions
def process_new_email(email: Email):
    """Process a newly received email"""
    # Categorize
    email.category = categorizer.categorize(email)
    
    # Index in Elasticsearch
    search_service.index_email(email)
    
    # Store in vector DB
    config.email_collection.add(
        ids=[email.uid],
        documents=[email.body],
        metadatas=[{"category": email.category}]
    )
    
    # Notify if interested
    if email.category == "Interested":
        notifier.send_slack_notification(email)
        notifier.trigger_webhook(email)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)