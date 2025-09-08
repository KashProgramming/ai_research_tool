# AI-Powered Research Assistant

This tool is designed for retrieving information effortlessly. It's a user-friendly research assistant that allows users to enter links of various articles that they want content from. They can then ask questions related to the articles and the relevant content will be fetched and used by the AI to provide the user with an appropriate answer. There will also be sources provided to know exactly which link the answer was taken from.

## Live Demo
👉 Try it now: [**AI-Powered Research Assistant**](https://ai-powered-research-assistant.streamlit.app)

## Features

- Load URLs or upload text files with links to extract article content.
- Process extracted content using LangChain’s UnstructuredURLLoader for efficient analysis.
- Construct embedding vectors using HuggingFace embeddings and utilize ChromaDB, a high-performance vector database, for fast and effective information retrieval.
- Interact with a large language model by inputting queries and receiving answers along with source URLs for reference.


This research assistant streamlines the process of gathering and analyzing information, making it easier to extract key insights without having to manually go through multiple articles.
