![1513a619-6193-4a40-ae8d-f9019696df76 (1)](https://github.com/user-attachments/assets/182c042e-6379-49f1-b037-b3f10e545cca)

# ForensIQ: Elasticsearch and CSV Log Analysis with Ollama LLM AI

ForensIQ is an advanced log analysis tool that combines Elasticsearch with Ollama's AI capabilities to provide progressive summarization and entity tracking for forensic investigations.

## Key Features

- **AI-Powered Analysis**: Utilizes Ollama's LLMs (including Mistral, Llama3, and Mixtral) for intelligent log analysis
- **Progressive Summarization**: Builds context across large log sets with chunked processing
- **Entity Tracking**: Identifies and tracks entities (IPs, users, URLs) across logs
- **Threat Detection**: Maps suspicious activity to MITRE ATT&CK techniques
- **Dynamic Chunking**: Automatically adjusts log chunk sizes based on model context limits
- **Interactive Investigation**: Supports follow-up questions with full context awareness

## Supported Models

ForensIQ supports multiple Ollama models with different capabilities:

| Model        | RAM Required | Context Length | Best For                  |
|--------------|-------------|----------------|---------------------------|
| gemma:2b     | 2GB         | 8k tokens      | Quick analysis            |
| phi          | 2GB         | 2k tokens      | Compact analysis          |
| phi3:mini    | 2-3GB       | 128k tokens    | Long-context lightweight  |
| tinyllama    | 1GB         | 2k tokens      | Basic tasks               |
| llama2:7b    | 8GB         | 4-8k tokens    | Balanced performance      |
| mistral      | 4GB         | 32k tokens     | High quality output       |
| llama3:8b    | 12GB        | 8k tokens      | Improved reasoning        |
| llama3:70b   | 39GB        | 8k tokens      | High capability analysis  |
| mixtral      | 29GB        | 32k tokens     | Best overall quality      |

## Installation

1. Ensure you have Python 3.8+ installed
2. Install required packages:
   ```bash
   pip install requests yaspin psutil
3. Install required packages:

## Configuration
On first run, ForensIQ will create a configuration file (ollama-config.txt) with:

Elasticsearch URL and credentials

Ollama URL and selected model

Analysis parameters

The tool will automatically validate your system's RAM and suggest appropriate models.

##  Usage
   ```bash
   python forensiq.py
```
1. Select an Elasticsearch index to analyze
2. Choose between default or custom analysis prompts
3. Review the progressive analysis as it processes logs
4. Ask follow-up questions with full context awareness
5. Export complete reports as HTML

## Analysis Capabilities
ForensIQ performs comprehensive log analysis including:

- Suspicious event categorization
- Significance explanation
- Attacker objective identification
- Investigation step recommendations
- Defensive strategies
- Timeline reconstruction
- IOC extraction

## Output Format
Each analysis includes structured data extraction in JSON format:

json
{
  "entities": {
    "ip": ["1.2.3.4"],
    "user": ["admin"]
  },
  "themes": {
    "brute_force": ["multiple failed logins"]
  },
  "timeline": [
    {"timestamp": "2023-01-01T00:00:00Z", "event": "First failed login"}
  ],
  "attacks": {
    "T1110 - Brute Force": ["10 failed login attempts"]
  }
}
## Requirements
Elasticsearch cluster (7.x+)

Ollama server with at least one LLM model

Minimum 2GB RAM (4GB+ recommended)

Python 3.8+

## License
ForensIQ is released under the MIT License.
