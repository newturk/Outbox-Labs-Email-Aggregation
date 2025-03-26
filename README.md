# Outbox-Labs-Email-Aggregation
Outbox Labs(Assignment)
# ReachInbox Email Aggregator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A feature-rich onebox email aggregator with real-time synchronization, AI categorization, and smart reply suggestions.

## Features

- 📨 Real-time IMAP email synchronization (IDLE mode)
- 🔍 Powerful search with Elasticsearch backend
- 🤖 AI-powered email categorization
- 🔔 Slack & webhook notifications for interested leads
- ✨ AI-generated reply suggestions (RAG model)
- 🖥️ Streamlit web interface

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- API keys for:
  - OpenAI
  - Slack (optional)
  - Webhook.site (optional)

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reachinbox-email-aggregator.git
   cd reachinbox-email-aggregator

2. Create environment file:
   ```bash
   cp .env.example .env
  Edit the .env file with your credentials.

3. 

