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

![image](https://github.com/user-attachments/assets/801060cd-4e09-431c-b6e9-e4a496751e74)
![image](https://github.com/user-attachments/assets/83c52992-1c7b-4ae6-be76-0a5da608d015)
   
### Final Analysis Report
   ```
   1. Executive Summary:
An unauthorized attacker has gained access to the network and compromised several systems, including IEWIN7 and LAPTOP-JU4M3I0E. The attacker established persistence by creating a local account (support) with administrative privileges on LAPTOP-JU4M3I0E and potentially exfiltrating sensitive data using the WMIGhost script.

2. Detailed Timeline of Events:
- 2023-01-01T05:54.9Z: Creation of local account support on LAPTOP-JU4M3I0E with administrative privileges
- 2023-01-01T05:55.0Z: Addition of new account to admin group on LAPTOP-JU4M3I0E
- 2023-01-01T05:55.1Z: Modification of WMI filters on IEWIN7

3. Identified Attack Patterns Mapped to MITRE ATT&CK:
- T1078 - Lateral Movement: Creation and modification of local accounts, addition to admin group
- T1575 - Tool Use (WMIGhost): Potential data exfiltration

4. Recommended Investigation Steps:
- Analyze the WMIGhost script for any indicators of compromise (IOCs) such as command-and-control (C2) servers or malicious payloads.
- Investigate the account creation event to determine if it was a result of an attack or a legitimate administrative action.
- Examine the local administrator group on LAPTOP-JU4M3I0E for any unusual additions or modifications.
- Monitor network traffic for outbound connections to suspicious IP addresses or domains associated with the WMIGhost script.

5. Mitigation Strategies:
- Implement strong password policies and enforce multi-factor authentication for administrative accounts.
- Regularly review and monitor local administrator groups for any unauthorized additions or modifications.
- Restrict the use of scripts like WMIGhost where possible, and implement strict access controls to limit their execution.
- Implement an endpoint detection and response (EDR) solution to detect and respond to such activities in real-time.

6. List of all Critical IOCs:
- IPs: None identified
- Users: PC04\\IEUser, NT AUTHORITY\\SYSTEM
- URLs: http://kumardeep.sosblogs.com/The-first-blog-b1/RSS-b1-rss2-posts.htm, http://blogs.rediff.com/anilchopra/feed/, http://www.blogster.com/kapoorsunil09/profile/rss

7. Any Remaining Questions or Areas Needing Clarification:
- The purpose of the account creation event on LAPTOP-JU4M3I0E and whether it was a result of an attack or a legitimate administrative action.
- The origin and intent behind the WMIGhost script execution on IEWIN7.
```
### Follow-up Response #1
```
 Based on the provided logs and analysis summary, here are some key points to address your question:

1. Previously identified entities or patterns: The IP address `10.0.2.16` and user account `S-1-5-20` have been associated with a lateral movement attack using Remote Desktop Protocol (RDP).

2. Timeline events related to the question: The suspicious activity started at approximately 23:23, as indicated in the event logs. Multiple connections from IP address `10.0.2.16` to system `MSEDGEWIN10` were observed over a short period of time.

3. Attack techniques that may be relevant: The attacker appears to be using RDP for lateral movement, attempting to establish persistent access or perform reconnaissance on the targeted system.

4. Any new insights from connecting the question to existing data: If further investigation reveals that the user account `S-1-5-20` has elevated privileges, it could indicate that the attacker gained unauthorized access to sensitive systems or data. Additionally, if any unusual activity is found on other systems accessed from IP address `10.0.2.16`, it may suggest a broader campaign by the attacker.

In light of these findings, it would be advisable to:
- Investigate the user account `S-1-5-20` and its privileges.
- Examine other systems that have been accessed from IP address `10.0.2.16`.
- Check for any unusual activity on the system `MSEDGEWIN10`, such as file modifications, process executions, or network connections.
- Investigate the origin of IP address `10.0.2.16` and its associated subnet to determine if it is part of a larger campaign.

Defensive recommendations:
- Implement strong password policies and enforce multi-factor authentication for RDP sessions.
- Regularly update and patch all systems, especially those exposed to the internet.
- Disable RDP if it is not necessary for business operations.
- Use a VPN or bastion host for remote access to internal networks.
- Monitor network traffic for unusual patterns of RDP usage.
- Implement an intrusion detection system (IDS) or intrusion prevention system (IPS) to detect and prevent such attacks.
```

## Requirements
Elasticsearch cluster (7.x+)

Ollama server with at least one LLM model

Minimum 2GB RAM (4GB+ recommended)

Python 3.8+

## License
ForensIQ is released under the MIT License.
