#!/usr/bin/env python3

##### Some good LLM's are Mistral (basic 4GB) llama3:70b (about 39GB) and there is mixtral (29GB) which is similar to mistral but with better quality

"""
Enhanced Elasticsearch + Ollama Log Analyzer with progressive summarization and entity tracking
"""
import requests
import json
import sys
import os
from requests.auth import HTTPBasicAuth
import urllib3
from datetime import datetime
from yaspin import yaspin
import time
from collections import defaultdict
import re
import psutil

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("")
print("Developed by Jacob Wilson - Version 0.1")
print("dfirvault@gmail.com")
print("")

# =============== CONFIGURATION ===============
CONFIG_FILE = "ollama-config.txt"

# Define model options with descriptions
MODEL_OPTIONS = {
    "gemma:2b": "gemma:2b (Very fast, ~2GB RAM, ideal for quick analysis, ~8k tokens)",
    "phi": "phi-2 (Compact and efficient, ~2GB RAM, ~2k tokens)",
    "phi3:mini": "phi3:mini (Long-context lightweight model, ~2–3GB RAM, 128k tokens)",
    "tinyllama": "tinyllama (Extremely small footprint, ~1GB RAM, basic tasks, ~2k tokens)",
    "llama2:7b": "llama2:7b (Good balance of speed and performance, ~8GB RAM, ~4k–8k tokens)",
    "mistral": "mistral (High quality output, ~4GB RAM, ~8k tokens)",
    "llama3:8b": "llama3:8b (Improved reasoning, ~12GB RAM, ~8k tokens)",
    "llama3:70b": "llama3:70b (High capability, requires ~39GB VRAM, ~8k tokens)",
    "mixtral": "mixtral (Best output, large RAM/VRAM footprint ~29GB, ~32k tokens)"
}

MODEL_CONTEXT_LENGTHS = {
    "gemma:2b": 8192,
    "phi": 2048,
    "phi3:mini": 128000,
    "tinyllama": 2048,
    "llama2:7b": 4096,
    "mistral": 32768,
    "llama3:8b": 8192,
    "llama3:70b": 8192,
    "mixtral": 32768
}

MODEL_RAM_REQUIREMENTS_GB = {
    "tinyllama": 1,
    "gemma:2b": 2,
    "phi": 2,
    "phi-3-mini": 3,
    "mistral": 4,
    "llama2:7b": 8,
    "llama3:8b": 12,
    "mixtral": 29,
    "llama3:70b": 39
}

# Required configuration keys and their validation functions
REQUIRED_CONFIG = {
    'ELASTICSEARCH_URL': {
        'validate': lambda x: x.startswith(('http://', 'https://')),
        'error_msg': "must start with http:// or https://"
    },
    'ELASTIC_USERNAME': {
        'validate': lambda x: len(x) > 0,
        'error_msg': "cannot be empty"
    },
    'ELASTIC_PASSWORD': {
        'validate': lambda x: len(x) > 0,
        'error_msg': "cannot be empty"
    },
    'OLLAMA_URL': {
        'validate': lambda x: x.startswith(('http://', 'https://')),
        'error_msg': "must start with http:// or https://"
    },
    'OLLAMA_MODEL': {
        'validate': lambda x: x in MODEL_OPTIONS,
        'error_msg': f"must be one of: {', '.join(MODEL_OPTIONS.keys())}"
    }
}

def get_available_ram_gb():
    """Returns available system RAM in GB"""
    available_bytes = psutil.virtual_memory().available
    return round(available_bytes / (1024 ** 3), 1)

def validate_config(config):
    """Validate the configuration and prompt for missing/invalid values"""
    updated_config = config.copy()
    needs_save = False
    
    for key, validation in REQUIRED_CONFIG.items():
        value = config.get(key, '').strip()
        
        # If value is missing or invalid, prompt user
        while key not in config or not validation['validate'](value):
            if key in config:
                print(f"\n⚠️ Invalid value for {key} in config: {value}")
                print(f"   Requirement: {validation['error_msg']}")
            else:
                print(f"\n⚠️ Missing required configuration: {key}")
            
            # Special handling for OLLAMA_MODEL
            if 'OLLAMA_MODEL' in config:
                print(f"\nYou have chosen '{config['OLLAMA_MODEL']}' as the model to use.")
                ram_gb = get_available_ram_gb()
                print(f"Detected Available RAM: {ram_gb} GB")
                print("Press Enter to keep current, or select one of the following options:")

                sorted_models = list(MODEL_OPTIONS.items())
                for i, (model, desc) in enumerate(sorted_models, 1):
                    required_ram = MODEL_RAM_REQUIREMENTS_GB.get(model, 0)
                    suitable = ram_gb >= required_ram
                    status = "✓" if suitable else "✗"
                    print(f"{i}. {status} {desc}")

                choice = input(f"\nEnter your choice (1-{len(MODEL_OPTIONS)}) or press Enter to keep current: ").strip()
                if choice:
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(sorted_models):
                            selected_model = list(MODEL_OPTIONS.keys())[choice_idx]
                            if selected_model != config['OLLAMA_MODEL']:
                                config['OLLAMA_MODEL'] = selected_model
                                save_config(config)
                                print(f"Model changed to: {selected_model}")
                        else:
                            print("Invalid choice, keeping current model.")
                    except ValueError:
                        print("Invalid input, keeping current model.")
            else:
                value = input(f"Enter value for {key}: ").strip()
            
            # Validate the new value
            if validation['validate'](value):
                updated_config[key] = value
                needs_save = True
                break
            else:
                print(f"Invalid value: {validation['error_msg']}")
    
    # Ensure Ollama URL has the correct API path
    if 'OLLAMA_URL' in updated_config:
        ollama_url = updated_config['OLLAMA_URL']
        if not ollama_url.endswith('/api/generate'):
            updated_config['OLLAMA_URL'] = ollama_url.rstrip('/') + '/api/generate'
            needs_save = True
    
    return updated_config, needs_save

def save_config(config):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")
    print(f"✅ Configuration saved to {CONFIG_FILE}")

def load_or_create_config():
    """Load configuration from file or create it if it doesn't exist"""
    if os.path.exists(CONFIG_FILE):
        print(f"Loading configuration from {CONFIG_FILE}")
        config = {}
        try:
            with open(CONFIG_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()

            # Validate the loaded config
            config, needs_save = validate_config(config)
            if needs_save:
                save_config(config)

            # Show model selection prompt with RAM check and ✓/✗ marks
            if 'OLLAMA_MODEL' in config:
                print(f"\nYou have chosen '{config['OLLAMA_MODEL']}' as the model to use.")
                ram_gb = get_available_ram_gb()
                print(f"Detected Available RAM: {ram_gb} GB")
                print("Press Enter to keep current, or select one of the following options:")

                sorted_models = list(MODEL_OPTIONS.items())
                for i, (model, desc) in enumerate(sorted_models, 1):
                    required_ram = MODEL_RAM_REQUIREMENTS_GB.get(model, 0)
                    suitable = ram_gb >= required_ram
                    status = "✓" if suitable else "✗"
                    print(f"{i}. {status} {desc}")

                choice = input(f"\nEnter your choice (1-{len(MODEL_OPTIONS)}) or press Enter to keep current: ").strip()
                if choice:
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(sorted_models):
                            selected_model = list(MODEL_OPTIONS.keys())[choice_idx]
                            if selected_model != config['OLLAMA_MODEL']:
                                config['OLLAMA_MODEL'] = selected_model
                                save_config(config)
                                print(f"Model changed to: {selected_model}")
                        else:
                            print("Invalid choice, keeping current model.")
                    except ValueError:
                        print("Invalid input, keeping current model.")

            return config
        except Exception as e:
            print(f"⚠️ Error reading config file: {e}")
            print("Creating new configuration...")

    # Create new config
    print("Configuration file not found or invalid. Creating new config...")
    config = {}
    for key in REQUIRED_CONFIG:
        if key == 'OLLAMA_MODEL':
            ram_gb = get_available_ram_gb()
            print(f"\nDetected Available RAM: {ram_gb} GB")
            print("Select the Ollama model you want to use:")

            sorted_models = list(MODEL_OPTIONS.items())
            for i, (model, desc) in enumerate(sorted_models, 1):
                required_ram = MODEL_RAM_REQUIREMENTS_GB.get(model, 0)
                suitable = ram_gb >= required_ram
                status = "✓" if suitable else "✗"
                print(f"{i}. {status} {desc}")

            while True:
                choice = input(f"Enter your choice (1-{len(MODEL_OPTIONS)}): ").strip()
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(sorted_models):
                        config[key] = list(MODEL_OPTIONS.keys())[choice_idx]
                        print(f"Selected model: {config[key]}")
                        break
                    else:
                        print(f"Invalid choice. Please enter a number between 1 and {len(MODEL_OPTIONS)}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
        else:
            prompt = f"Enter value for {key}"
            if 'error_msg' in REQUIRED_CONFIG[key]:
                prompt += f" ({REQUIRED_CONFIG[key]['error_msg']})"
            prompt += ": "

            while True:
                value = input(prompt).strip()
                if REQUIRED_CONFIG[key]['validate'](value):
                    config[key] = value
                    break
                else:
                    print(f"Invalid value: {REQUIRED_CONFIG[key]['error_msg']}")

    # Ensure Ollama URL has the correct API path
    if not config['OLLAMA_URL'].endswith('/api/generate'):
        config['OLLAMA_URL'] = config['OLLAMA_URL'].rstrip('/') + '/api/generate'

    save_config(config)
    return config

# Load configuration
config = load_or_create_config()

# Set global variables from config
ELASTICSEARCH_URL = config['ELASTICSEARCH_URL']
ELASTIC_USERNAME = config['ELASTIC_USERNAME']
ELASTIC_PASSWORD = config['ELASTIC_PASSWORD']
OLLAMA_URL = config['OLLAMA_URL']
OLLAMA_MODEL = config['OLLAMA_MODEL']

# Other configuration parameters
SAMPLE_LOG_COUNT = 20
DEFAULT_TIME_WINDOW = 24  # hours
MAX_RETRIES = 5
BACKOFF_FACTOR = 2  # seconds, multiplied exponentially
TEMPERATURE_VALUE = 0.1  # 0.1 is default Lower for more deterministic forensic analysis
LOGS_PER_CHUNK = 10 #500 default must be low to stay under the model ram tocken limit, try 5 and monitor logs to make sure it's not truncating
LOG_CHUNK_SIZE = 200 #default 200
NUM_CTX_VALUE = MODEL_CONTEXT_LENGTHS.get(OLLAMA_MODEL, 8192)  # Default to 8192 if model not found
# =============================================

class AnalysisContext:
    def __init__(self):
        self.progressive_summary = ""
        self.entities = defaultdict(list)
        self.themes = defaultdict(list)
        self.timeline_events = []
        self.attacks = defaultdict(list)
    
    def update(self, chunk_summary, extracted_data):
        """Update the context with new information from a chunk"""
        self.progressive_summary += f"\n\n=== Chunk Analysis ===\n{chunk_summary}"
        
        # Update entities
        for entity_type, values in extracted_data.get('entities', {}).items():
            self.entities[entity_type].extend(values)
        
        # Update themes
        for theme, examples in extracted_data.get('themes', {}).items():
            self.themes[theme].extend(examples)
        
        # Update timeline
        self.timeline_events.extend(extracted_data.get('timeline', []))
        
        # Update attack techniques
        for technique, events in extracted_data.get('attacks', {}).items():
            self.attacks[technique].extend(events)
    
    def get_context_prompt(self):
        """Generate a prompt section with the current context"""
        context_prompt = "\n\n=== CURRENT ANALYSIS CONTEXT ==="
        
        if self.entities:
            context_prompt += "\n\nKey Entities Identified:"
            for entity_type, values in self.entities.items():
                unique_values = list(set(values))[:10]  # Show top 10 unique values
                context_prompt += f"\n- {entity_type.title()}s: {', '.join(unique_values)}"
                if len(values) > 10:
                    context_prompt += f" (and {len(values)-10} more)"
        
        if self.themes:
            context_prompt += "\n\nKey Themes Identified:"
            for theme, examples in self.themes.items():
                context_prompt += f"\n- {theme}: {len(examples)} occurrences"
        
        if self.timeline_events:
            context_prompt += "\n\nTimeline Highlights:"
            for event in sorted(self.timeline_events, key=lambda x: x.get('timestamp', ''))[:5]:
                context_prompt += f"\n- [{event.get('timestamp')}] {event.get('event')}"
        
        if self.attacks:
            context_prompt += "\n\nPotential ATT&CK Techniques:"
            for technique, events in self.attacks.items():
                context_prompt += f"\n- {technique}: {len(events)} related events"
        
        context_prompt += "\n\n=== END CURRENT CONTEXT ===\n"
        return context_prompt
    
    def get_summary_sample(self, max_length=2000):
        """Get a truncated version of the progressive summary"""
        if len(self.progressive_summary) <= max_length:
            return self.progressive_summary
        return self.progressive_summary[:max_length] + "... [truncated]"

def get_available_indices():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = requests.get(
                f"{ELASTICSEARCH_URL}/_cat/indices?format=json&h=index,docs.count,health",
                auth=HTTPBasicAuth(ELASTIC_USERNAME, ELASTIC_PASSWORD),
                verify=False,
                timeout=10
            )
            response.raise_for_status()
            return [
                (idx['index'], idx.get('docs.count', '?'), idx.get('health', '?'))
                for idx in response.json()
                if not idx['index'].startswith('.')  # Skip system indices
            ]
        except requests.exceptions.RequestException as e:
            retries += 1
            wait = BACKOFF_FACTOR ** retries
            print(f"⚠️ Failed to get indices: {e}. Retry {retries}/{MAX_RETRIES} in {wait}s...")
            time.sleep(wait)

    print("⛔ Max retries reached. Could not retrieve indices.")
    return None

def display_index_menu(indices):
    print("\nAvailable Indices (index: documents | health):")
    for i, (index, count, health) in enumerate(indices, 1):
        print(f"{i}. {index}: {count} docs | {health}")
    
    while True:
        try:
            choice = input("\nSelect an index (number), 'm' for mapping, or 0 to exit: ")
            if choice == '0':
                return None, None
            elif choice.lower() == 'm':
                idx_choice = int(input("Enter index number to view mapping: "))
                if 1 <= idx_choice <= len(indices):
                    return None, indices[idx_choice-1][0]
            else:
                choice = int(choice)
                if 1 <= choice <= len(indices):
                    return indices[choice-1][0], None
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number, 'm', or 0.")

def get_index_mapping(index):
    try:
        response = requests.get(
            f"{ELASTICSEARCH_URL}/{index}/_mapping",
            auth=HTTPBasicAuth(ELASTIC_USERNAME, ELASTIC_PASSWORD),
            verify=False,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"⛔ Failed to get mapping: {e}")
        return None

def calculate_prompt_size(prompt_text):
    """Estimate token count using a simple heuristic (4 chars ≈ 1 token)"""
    return len(prompt_text) // 4

def find_optimal_chunk_size(base_prompt_intro, logs, timestamp_field, initial_chunk_size, max_context_tokens):
    """Dynamically adjust chunk size to fit within model's context window"""
    chunk_size = initial_chunk_size
    min_chunk_size = 5  # Don't go below this
    
    def format_chunk(chunk):
        return "\n".join(
            f"[{log.get(timestamp_field, 'no timestamp')}] " + 
            " ".join(f"{k}={v}" for k, v in log.items() if k != timestamp_field)
            for log in chunk
        )
    
    while chunk_size >= min_chunk_size:
        # Test with the first chunk
        test_chunk = logs[:chunk_size]
        chunk_logs_formatted = format_chunk(test_chunk)
        
        # Build the test prompt
        test_prompt = f"""{base_prompt_intro}

=== NEW LOGS TO ANALYZE ===
{chunk_logs_formatted}
=== END NEW LOGS ==="""
        
        # Estimate token usage
        total_tokens = calculate_prompt_size(test_prompt)
        
        # Leave 20% headroom for the model's response
        if total_tokens <= (max_context_tokens * 0.8):
            print(f"✅ Validated chunk size: {chunk_size} (estimated tokens: {total_tokens}/{max_context_tokens})")
            return chunk_size
        else:
            #print(f"⚠️ Chunk size {chunk_size} too large (estimated tokens: {total_tokens}/{max_context_tokens})")
            new_chunk_size = int(chunk_size * 0.9)  # Reduce by 10%
            if new_chunk_size == chunk_size:  # Prevent infinite loop
                new_chunk_size -= 1
            chunk_size = max(min_chunk_size, new_chunk_size)
            #print(f"Trying smaller chunk size: {chunk_size}")
    
    print(f"⚠️ Could not find suitable chunk size, using minimum: {min_chunk_size}")
    return min_chunk_size

def detect_timestamp_field(mapping_data):
    try:
        index_name = next(iter(mapping_data.keys()))
        properties = mapping_data[index_name]['mappings']['properties']
        
        # Common timestamp field names (case insensitive)
        timestamp_candidates = [
            '@timestamp', 'timestamp', 'time', 'event_time',
            'created_at', 'log_time', 'date_time', 'datetime'
        ]
        
        # Known numeric fields that should NEVER be treated as timestamps
        excluded_fields = [
            'eventid', 'event_id', 'id', 'record_number',
            'process_id', 'pid', 'thread_id', 'message_id'
        ]
        
        # Check candidates first
        for field in timestamp_candidates:
            if field.lower() in properties:
                field_type = properties[field.lower()].get('type')
                if field_type in ('date', 'date_nanos'):
                    return field.lower()
        
        # Fallback to any date-type field not in excluded list
        date_fields = [
            field for field, props in properties.items()
            if props.get('type') in ('date', 'date_nanos')
            and field.lower() not in [x.lower() for x in excluded_fields]
        ]
        
        return date_fields[0] if date_fields else None
        
    except Exception:
        return None

def convert_epoch_to_iso8601(log_entry):
    def is_epoch(value):
        return isinstance(value, (int, float)) and 1000000000 < value < 9999999999999

    def convert(value):
        try:
            if isinstance(value, (int, float)):
                if value > 1e12:
                    dt = datetime.utcfromtimestamp(value / 1000)
                else:
                    dt = datetime.utcfromtimestamp(value)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
        return value

    if isinstance(log_entry, dict):
        return {
            k: convert_epoch_to_iso8601(v) if isinstance(v, (dict, list)) else convert(v)
            for k, v in log_entry.items()
        }
    elif isinstance(log_entry, list):
        return [convert_epoch_to_iso8601(i) for i in log_entry]
    else:
        return convert(log_entry)

def get_total_docs_count(index):
    try:
        response = requests.get(
            f"{ELASTICSEARCH_URL}/_cat/indices/{index}?format=json&h=docs.count",
            auth=HTTPBasicAuth(ELASTIC_USERNAME, ELASTIC_PASSWORD),
            verify=False,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0:
            return int(data[0]['docs.count'])
        else:
            return None
    except Exception as e:
        print(f"⚠️ Failed to get total docs count: {e}")
        return None

def get_logs_from_index(index, hours=DEFAULT_TIME_WINDOW):
    total_docs = get_total_docs_count(index)
    if total_docs is None:
        total_docs = '?'
    try:
        mapping = get_index_mapping(index)
        timestamp_field = None

        if mapping:
            timestamp_field = detect_timestamp_field(mapping)
            index_name = next(iter(mapping.keys()))
            props = mapping[index_name]['mappings'].get('properties', {})
            if not timestamp_field or timestamp_field not in props:
                print(f"⚠️ Timestamp field '{timestamp_field}' not found in mapping properties.")
                timestamp_field = None

        print(f"\nℹ️ Index analysis:")
        print(f"- Detected timestamp field: {timestamp_field or 'Not found'}")

        if timestamp_field:
            query = {
                "size": 1000,
                "sort": [{timestamp_field: {"order": "desc"}}],
                "query": {
                    "match_all": {}
                }
            }
        else:
            query = {
                "size": 1000,
                "query": {"match_all": {}}
            }

        response = requests.post(
            f"{ELASTICSEARCH_URL}/{index}/_search?scroll=2m",
            auth=HTTPBasicAuth(ELASTIC_USERNAME, ELASTIC_PASSWORD),
            json=query,
            verify=False,
            timeout=30
        )

        if response.status_code != 200:
            try:
                error_info = response.json()
            except Exception:
                error_info = response.text
            print(f"⛔ Elasticsearch returned error:\n{error_info}")
            response.raise_for_status()

        data = response.json()
        total_hits = data['hits']['total']['value'] if isinstance(data['hits']['total'], dict) else data['hits']['total']
        
        all_logs = []
        scroll_id = data.get('_scroll_id')
        hits = data['hits']['hits']
        all_logs.extend([convert_epoch_to_iso8601(hit['_source']) for hit in hits])
        
        last_reported = 0
        report_interval = 1000
        print(f"\rRetrieved {len(all_logs)}/{total_docs} logs so far...", end='', flush=True)
        while hits and scroll_id:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    scroll_response = requests.post(
                        f"{ELASTICSEARCH_URL}/_search/scroll",
                        auth=HTTPBasicAuth(ELASTIC_USERNAME, ELASTIC_PASSWORD),
                        json={"scroll": "2m", "scroll_id": scroll_id},
                        verify=False,
                        timeout=30
                    )
                    scroll_response.raise_for_status()
                    scroll_data = scroll_response.json()
                    hits = scroll_data['hits']['hits']
                    scroll_id = scroll_data.get('_scroll_id')
                    all_logs.extend([convert_epoch_to_iso8601(hit['_source']) for hit in hits])
                    
                    if len(all_logs) >= last_reported + report_interval or len(all_logs) == total_hits:
                        print(f"\rRetrieved {len(all_logs)} logs so far...", end='', flush=True)
                        last_reported = len(all_logs)
                    
                    break
                except requests.exceptions.RequestException as e:
                    retries += 1
                    wait = BACKOFF_FACTOR ** retries
                    print(f"⚠️ Network error during scroll request: {e}. Retry {retries}/{MAX_RETRIES} in {wait}s...")
                    time.sleep(wait)
            else:
                print("⛔ Max retries reached during scroll. Aborting log retrieval.")
                break
        print()
        print(f"✅ Retrieved {len(all_logs)}/{total_hits} total logs")
        if len(all_logs) == 10000:
            print("⚠️ Note: Exactly 10,000 logs retrieved. This may indicate a result limit.")
            print("ℹ️ To retrieve more, consider increasing 'index.max_result_window' in Elasticsearch.")
        return all_logs, timestamp_field

    except Exception as e:
        print(f"\n⛔ Error retrieving logs: {type(e).__name__}: {e}")
        return None, None

def chunk_logs(logs, chunk_size=LOG_CHUNK_SIZE):
    """Yield successive chunks of logs."""
    for i in range(0, len(logs), chunk_size):
        yield logs[i:i + chunk_size]

def extract_entities_from_response(response_text):
    """Parse the LLM response to extract entities, themes, and timeline events"""
    entities = defaultdict(list)
    themes = defaultdict(list)
    timeline = []
    attacks = defaultdict(list)
    
    # First try to find and parse JSON blocks
    json_blocks = []
    stack = []
    start_idx = -1
    
    # Scan for potential JSON blocks
    for i, c in enumerate(response_text):
        if c == '{':
            if not stack:
                start_idx = i
            stack.append(c)
        elif c == '}':
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    json_blocks.append((start_idx, i+1))
                    start_idx = -1
    
    # Try parsing each JSON block until we find a valid one
    extracted_data = None
    for start, end in json_blocks:
        json_str = response_text[start:end]
        try:
            # Clean common JSON issues
            json_str = json_str.replace('\\"', '"')  # Fix escaped quotes
            json_str = json_str.replace("\\'", "'")  # Fix escaped single quotes
            json_str = re.sub(r'\\[^"/bfnrtu]', r'\\\\', json_str)  # Fix invalid escapes
            
            extracted = json.loads(json_str)
            if isinstance(extracted, dict):
                extracted_data = extracted
                break
        except json.JSONDecodeError as e:
            continue
        except Exception as e:
            continue
    
    if extracted_data:
        # Safely handle entities
        if 'entities' in extracted_data:
            entities_data = extracted_data['entities']
            if isinstance(entities_data, dict):
                for entity_type, values in entities_data.items():
                    if isinstance(values, list):
                        entities[entity_type].extend(values)
                    elif isinstance(values, str):
                        entities[entity_type].append(values)
        
        # Safely handle themes
        if 'themes' in extracted_data:
            themes_data = extracted_data['themes']
            if isinstance(themes_data, dict):
                for theme, examples in themes_data.items():
                    if isinstance(examples, list):
                        themes[theme].extend(examples)
                    elif isinstance(examples, str):
                        themes[theme].append(examples)
        
        # Safely handle timeline
        if 'timeline' in extracted_data:
            timeline_data = extracted_data['timeline']
            if isinstance(timeline_data, list):
                timeline.extend(timeline_data)
        
        # Safely handle attacks
        if 'attacks' in extracted_data:
            attacks_data = extracted_data['attacks']
            if isinstance(attacks_data, dict):
                for technique, events in attacks_data.items():
                    if isinstance(events, list):
                        attacks[technique].extend(events)
                    elif isinstance(events, str):
                        attacks[technique].append(events)
        
        return {
            'entities': dict(entities),
            'themes': dict(themes),
            'timeline': timeline,
            'attacks': dict(attacks)
        }
    
    # Fallback to text parsing if JSON not found or parsing failed
    lines = response_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect sections
        if line.startswith('Entities:'):
            current_section = 'entities'
            continue
        elif line.startswith('Themes:'):
            current_section = 'themes'
            continue
        elif line.startswith('Timeline:'):
            current_section = 'timeline'
            continue
        elif line.startswith('ATT&CK Techniques:'):
            current_section = 'attacks'
            continue
        
        # Parse based on current section
        if current_section == 'entities' and ':' in line:
            try:
                entity_type, values = line.split(':', 1)
                entity_type = entity_type.strip().lower()
                values = [v.strip() for v in values.split(',') if v.strip()]
                entities[entity_type].extend(values)
            except:
                continue
        elif current_section == 'themes' and ':' in line:
            try:
                theme, desc = line.split(':', 1)
                themes[theme.strip().lower()].append(desc.strip())
            except:
                continue
        elif current_section == 'timeline' and ']' in line:
            try:
                parts = line.split(']', 1)
                timestamp = parts[0][1:].strip()
                event = parts[1].strip()
                timeline.append({'timestamp': timestamp, 'event': event})
            except:
                continue
        elif current_section == 'attacks' and ':' in line:
            try:
                technique, desc = line.split(':', 1)
                attacks[technique.strip()].append(desc.strip())
            except:
                continue
    
    return {
        'entities': dict(entities),
        'themes': dict(themes),
        'timeline': timeline,
        'attacks': dict(attacks)
    }

def show_default_prompt():
    default_prompt = """Default Prompt:
You are a DFIR analyst. Analyze these logs with:
1. Categorization of suspicious events
2. Significance explanation
3. Attacker objectives
4. Investigation steps
5. Defensive recommendations

For each analysis, also extract and return the following in JSON format at the end of your response:
{
  "entities": {
    "ip": ["1.2.3.4", "5.6.7.8"],
    "user": ["admin", "system"],
    "url": ["example.com/path"]
  },
  "themes": {
    "brute_force": ["multiple failed logins for admin"],
    "data_exfiltration": ["large outbound transfer to unknown IP"]
  },
  "timeline": [
    {"timestamp": "2023-01-01T00:00:00Z", "event": "First failed login attempt"},
    {"timestamp": "2023-01-01T00:05:00Z", "event": "Successful login with default credentials"}
  ],
  "attacks": {
    "T1110 - Brute Force": ["10 failed login attempts for admin user"],
    "T1048 - Exfiltration": ["500MB transfer to external IP"]
  }
}"""
    print("\n" + "="*50)
    print(default_prompt)
    print("="*50 + "\n")

def save_html_report(analysis_history, index_name=None):
    try:
        from tkinter import Tk, filedialog
        import os
        
        root = Tk()
        root.withdraw()  # Hide the main window
        
        # Create HTML content
        index_title = f" - {index_name}" if index_name else ""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Log Analysis Report{index_title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0 auto; max-width: 900px; padding: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                h1 {{ border-bottom: 2px solid #3498db; }}
                h2 {{ border-bottom: 1px solid #ddd; margin-top: 30px; }}
                pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
                .response {{ margin-bottom: 30px; }}
                .index-info {{ color: #27ae60; font-weight: bold; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1>Log Analysis Report{index_title}</h1>
            <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            {f'<div class="index-info">Analyzed Index: {index_name}</div>' if index_name else ''}
        """
        
        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML Files", "*.html"), ("All Files", "*.*")],
            title="Save Analysis Report"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\n✅ Report saved successfully to: {os.path.abspath(file_path)}")
        else:
            print("\n⚠️ Report saving cancelled.")
            
    except ImportError:
        print("\n⚠️ Tkinter not available. Please specify a filename to save the report:")
        file_path = input("Enter file path (e.g., report.html): ").strip()
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"\n✅ Report saved successfully to: {file_path}")
        else:
            print("\n⚠️ Report saving cancelled.")
    except Exception as e:
        print(f"\n⛔ Failed to save report: {e}")

def clean_analysis_output(text):
    # Fix eventID misinterpreted as timestamps
    text = re.sub(
        r'EventID (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)',
        lambda m: f"EventID {int(datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%SZ').timestamp())}",
        text
    )
    
    # Fix other common numeric fields
    text = re.sub(
        r'(ProcessID|ThreadID|MessageID) (\d{4}-\d{2}-\d{2})',
        lambda m: f"{m.group(1)} {m.group(2).replace('-', '')}",
        text
    )
    return text

def get_summary_sample(self, max_length=2000):
    """Get a truncated version of the progressive summary"""
    if len(self.progressive_summary) <= max_length:
        return self.progressive_summary
    return self.progressive_summary[:max_length] + "... [truncated]"

def analyze_with_ollama_chunked(logs, timestamp_field=None, index_name=None):
    if not logs:
        return None
    # Track chunk processing times
    chunk_times = []
    last_chunk_time = None
    def clean_log(log):
        return {
            k: v for k, v in log.items()
            if v not in (None, "", "none", "None") and str(v).lower() != "none"
        }
    global LOG_CHUNK_SIZE
    def calculate_average_chunk_time():
        if len(chunk_times) < 2:
            return None
        total_time = sum(chunk_times)
        return total_time / (len(chunk_times) - 1)

    def get_estimated_completion(current_chunk, total_chunks):
        avg_time = calculate_average_chunk_time()
        if avg_time is None:
            return "Estimating..."
        remaining = total_chunks - current_chunk
        estimated_seconds = remaining * avg_time
        if estimated_seconds < 60:
            return f"{int(estimated_seconds)} seconds"
        elif estimated_seconds < 3600:
            return f"{int(estimated_seconds / 60)} minutes"
        else:
            return f"{int(estimated_seconds / 3600)} hours {int((estimated_seconds % 3600)/60)} minutes"
    def format_logs(log_list):
        formatted = []
        for log in log_list:
            # Skip known numeric fields from timestamp processing
            log_copy = {k:v for k,v in log.items() 
                       if k.lower() not in ['eventid', 'event_id']}
            
            # Format the remaining log
            ts = log.get(timestamp_field, 'no timestamp')
            if isinstance(ts, (int, float)) and ts > 1e9:  # Likely epoch time
                ts = datetime.utcfromtimestamp(ts).isoformat()
                
            log_str = f"[{ts}] " + " ".join(
                f"{k}={v}" for k, v in clean_log(log_copy).items() 
                if k != timestamp_field
            )
            formatted.append(log_str)
        return "\n".join(formatted)

    chunks = [logs[i:i + LOGS_PER_CHUNK] for i in range(0, len(logs), LOGS_PER_CHUNK)]
    print(f"\nℹ️ Total logs: {len(logs)} split into {len(chunks)} chunk(s) for analysis.")

    context = AnalysisContext()
    all_formatted_logs = []
    analysis_history = []  # To store all analysis responses for the report

    def show_default_prompt():
        default_prompt = """Default Prompt:
You are a DFIR analyst. Analyze these logs with:
1. Categorization of suspicious events
2. Significance explanation
3. Attacker objectives
4. Investigation steps
5. Defensive recommendations

For each analysis, also extract and return the following in JSON format at the end of your response:
{
  "entities": {
    "ip": ["1.2.3.4", "5.6.7.8"],
    "user": ["admin", "system"],
    "url": ["example.com/path"]
  },
  "themes": {
    "brute_force": ["multiple failed logins for admin"],
    "data_exfiltration": ["large outbound transfer to unknown IP"]
  },
  "timeline": [
    {"timestamp": "2023-01-01T00:00:00Z", "event": "First failed login attempt"},
    {"timestamp": "2023-01-01T00:05:00Z", "event": "Successful login with default credentials"}
  ],
  "attacks": {
    "T1110 - Brute Force": ["10 failed login attempts for admin user"],
    "T1048 - Exfiltration": ["500MB transfer to external IP"]
  }
}"""
        print("\n" + "="*50)
        print(default_prompt)
        print("="*50 + "\n")

    def save_html_report(analysis_history, index_name=None):
        try:
            from tkinter import Tk, filedialog
            import os
            
            root = Tk()
            root.withdraw()  # Hide the main window
            
            # Create HTML content
            index_title = f" - {index_name}" if index_name else ""
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Log Analysis Report {index_title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0 auto; max-width: 900px; padding: 20px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    h1 {{ border-bottom: 2px solid #3498db; }}
                    h2 {{ border-bottom: 1px solid #ddd; margin-top: 30px; }}
                    pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                    .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
                    .response {{ margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <h1>Log Analysis Report {index_title}</h1>
                <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                {f'<div class="index-info">Analyzed Index: {index_name}</div>' if index_name else ''}
            """
            
            # Add each analysis response to the report
            for i, response in enumerate(analysis_history):
                if i == 0:
                    html_content += f"""
                    <div class="response">
                        <h2>Final Analysis Report</h2>
                        <pre>{response}</pre>
                    </div>
                    """
                else:
                    html_content += f"""
                    <div class="response">
                        <h2>Follow-up Response #{i}</h2>
                        <pre>{response}</pre>
                    </div>
                    """
            
            html_content += """
            </body>
            </html>
            """
            
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".html",
                filetypes=[("HTML Files", "*.html"), ("All Files", "*.*")],
                title="Save Analysis Report"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"\n✅ Report saved successfully to: {os.path.abspath(file_path)}")
            else:
                print("\n⚠️ Report saving cancelled.")
                
        except ImportError:
            print("\n⚠️ Tkinter not available. Please specify a filename to save the report:")
            file_path = input("Enter file path (e.g., report.html): ").strip()
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"\n✅ Report saved successfully to: {file_path}")
            else:
                print("\n⚠️ Report saving cancelled.")
        except Exception as e:
            print(f"\n⛔ Failed to save report: {e}")

    def send_prompt_to_ollama(prompt, is_followup=False, show_full_prompt=False):
        if show_full_prompt:
            print("\n" + "="*50)
            print("🧠 PROMPT TO OLLAMA")
            print("="*50)
            prompt_parts = prompt.split("=== NEW LOGS TO ANALYZE ===")
            print(prompt_parts[0].strip())
            print("[... LOG CONTENTS TRUNCATED ...]")
            print("="*50 + "\n")
        else:
            print("\n⏳ Processing chunk... (full prompt not shown)")

        start_time = time.time()
        spinner = yaspin(text="Analyzing logs...", color="yellow")
        spinner.start()

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": TEMPERATURE_VALUE,
                        "num_ctx": NUM_CTX_VALUE  # 131072 Explicitly set context window
                    }
                },
                timeout=None
            )
            response.raise_for_status()

            response_data = response.json()
            raw_response = response_data.get('response', '')
            
            # Clean common issues in the response
            cleaned_response = raw_response.replace('\\"', '"')
            cleaned_response = cleaned_response.replace('\\n', '\n')
            cleaned_response = re.sub(r'\\[^"/bfnrtu]', lambda m: f'\\{m.group(0)}', cleaned_response)

            elapsed = time.time() - start_time
            spinner.stop()
            print(f"\n✅ Analysis completed in {elapsed:.1f} seconds")

            return cleaned_response
        except Exception as e:
            spinner.stop()
            print(f"\n⛔ Ollama analysis failed after {time.time() - start_time:.1f} seconds: {e}")
            return None
        finally:
            spinner.stop()

    # === PROMPT CONSTRUCTION ===
    show_default_prompt()
    use_custom = input("\nWould you like to enter a custom prompt? (y/N): ").strip().lower()
    if use_custom == 'y':
        print("\nEnter your custom prompt (type END on a new line to finish):")
        custom_lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            custom_lines.append(line)
        base_prompt_intro = f"You are a DFIR analyst. {' '.join(custom_lines)}"
    else:
        base_prompt_intro = """You are a DFIR analyst. Analyze these logs with:
1. Categorization of suspicious events
2. Significance explanation
3. Attacker objectives
4. Investigation steps
5. Defensive recommendations

For each analysis, also extract and return the following in JSON format at the end of your response:
{
  "entities": {
    "ip": ["1.2.3.4", "5.6.7.8"],
    "user": ["admin", "system"],
    "url": ["example.com/path"]
  },
  "themes": {
    "brute_force": ["multiple failed logins for admin"],
    "data_exfiltration": ["large outbound transfer to unknown IP"]
  },
  "timeline": [
    {"timestamp": "2023-01-01T00:00:00Z", "event": "First failed login attempt"},
    {"timestamp": "2023-01-01T00:05:00Z", "event": "Successful login with default credentials"}
  ],
  "attacks": {
    "T1110 - Brute Force": ["10 failed login attempts for admin user"],
    "T1048 - Exfiltration": ["500MB transfer to external IP"]
  }
}"""
    # =====
     # === DYNAMIC CHUNK SIZE CALCULATION ===
    initial_chunk_size = LOG_CHUNK_SIZE  # Start with configured value
    max_context_tokens = MODEL_CONTEXT_LENGTHS.get(OLLAMA_MODEL, 8192)
    
    print(f"\n🔍 Calculating optimal chunk size (model context: {max_context_tokens} tokens)...")
    optimal_chunk_size = find_optimal_chunk_size(
        base_prompt_intro,
        logs,
        timestamp_field,
        initial_chunk_size,
        max_context_tokens
    )
    
    # Override the global LOG_CHUNK_SIZE for this analysis
    LOG_CHUNK_SIZE = optimal_chunk_size
    chunks = [logs[i:i + LOG_CHUNK_SIZE] for i in range(0, len(logs), LOG_CHUNK_SIZE)]
    print(f"\nℹ️ Total logs: {len(logs)} split into {len(chunks)} chunk(s) of size {LOG_CHUNK_SIZE} for analysis.")
    # =====
    # === CHUNK PROCESSING ===
    for i, chunk in enumerate(chunks, 1):
        chunk_start_time = time.time()
        if last_chunk_time is not None:
            chunk_times.append(chunk_start_time - last_chunk_time)
        last_chunk_time = chunk_start_time
        
        chunk_logs_formatted = format_logs(chunk)
        all_formatted_logs.append(chunk_logs_formatted)

        chunk_prompt = f"""{base_prompt_intro}

{context.get_context_prompt()}

=== NEW LOGS TO ANALYZE ===
{chunk_logs_formatted}
=== END NEW LOGS ===

Analyze these new logs in the context of what we've learned so far.
Highlight any connections to previously identified entities or patterns.
Include any new findings in JSON format at the end."""
        chunk_prompt = clean_analysis_output(chunk_prompt)
        # Update the status message with estimated time
        est_completion = get_estimated_completion(i, len(chunks))
        print(f"\n⏳ Analyzing chunk {i}/{len(chunks)} (Est. completion in {est_completion})...")
        
        summary = send_prompt_to_ollama(chunk_prompt, show_full_prompt=(i == 1))
        
        if summary:
            extracted_data = extract_entities_from_response(summary)
            context.update(summary, extracted_data)
            print(f"\nℹ️ Updated context with {len(extracted_data.get('entities', {}))} entities and {len(extracted_data.get('themes', {}))} themes")

    # === FINAL ANALYSIS ===
    if context.progressive_summary:
        final_prompt = f"""{base_prompt_intro}

=== COMPLETE ANALYSIS CONTEXT ===
{context.get_context_prompt()}

=== FULL LOG SUMMARY ===
{context.progressive_summary}
=== END SUMMARY ===

Generate a comprehensive final report with:
1. Executive summary of key findings
2. Detailed timeline of events
3. Identified attack patterns mapped to MITRE ATT&CK
4. Recommended investigation steps
5. Mitigation strategies
6. List of all critical IOCs (IPs, users, URLs)
7. Any remaining questions or areas needing clarification"""

        print("\n⏳ Generating final consolidated report with full context...")
        analysis = send_prompt_to_ollama(final_prompt, show_full_prompt=False)
        
        if analysis:
            print("\n" + "="*50)
            print(f"FINAL ANALYSIS REPORT FOR INDEX: {index_name}")
            print("="*50 + "\n")
            print(analysis)
            print("\n" + "="*50 + "\n")
            analysis_history.append(analysis)  # Store the final analysis

    # === FOLLOW-UP QUESTIONS ===
    while True:
        again = input("\nWould you like to ask another question? (y/N): ").strip().lower()
        if again != 'y':
            # Offer to save report when user says no to more questions
            if analysis_history:
                save_report = input("\nWould you like to export the complete report as HTML? (y/N): ").strip().lower()
                if save_report == 'y':
                    save_html_report(analysis_history, index_name)
            break

        print("\nEnter your follow-up (type END to finish):")
        followup_lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END':
                break
            followup_lines.append(line)
        
        # Create a new prompt that includes:
        # 1. The original analysis context
        # 2. A sample of the original logs (not all to avoid token limits)
        # 3. The progressive summary
        sample_logs = "\n".join(all_formatted_logs[:20])  # Show first 20 logs as sample
        
        followup_prompt = f"""You are a DFIR analyst continuing your investigation. Answer this question:
{' '.join(followup_lines)}

=== ANALYSIS CONTEXT ===
{context.get_context_prompt()}

=== SAMPLE OF ORIGINAL LOGS ===
{sample_logs}
[showing 20 of {len(all_formatted_logs)} total logs]

=== PROGRESSIVE ANALYSIS SUMMARY ===
{context.get_summary_sample()}... [truncated]

Please answer the question using this full context, highlighting any relevant:
1. Previously identified entities or patterns
2. Timeline events related to the question
3. Attack techniques that may be relevant
4. Any new insights from connecting the question to existing data"""
        
        print("\n⏳ Processing follow-up question with full context...")
        followup_response = send_prompt_to_ollama(followup_prompt, is_followup=True, show_full_prompt=False)
        
        if followup_response:
            print("\n" + "="*50)
            print(f"FOLLOW-UP RESPONSE FOR INDEX: {index_name}")
            print("="*50 + "\n")
            print(followup_response)
            print("\n" + "="*50 + "\n")
            analysis_history.append(followup_response)  # Store follow-up response
            
            # Extract any new entities from follow-up for context
            extracted_data = extract_entities_from_response(followup_response)
            context.update("Follow-up analysis:\n" + followup_response, extracted_data)

    return analysis_history[0] if analysis_history else None

def main():
    print("\n" + "="*50)
    print("Enhanced Elasticsearch Log Analyzer with Progressive Analysis")
    print("="*50)

    try:
        health = requests.get(
            f"{ELASTICSEARCH_URL}/_cluster/health",
            auth=HTTPBasicAuth(ELASTIC_USERNAME, ELASTIC_PASSWORD),
            verify=False,
            timeout=5
        )
        if health.status_code != 200:
            print("⛔ Failed to connect to Elasticsearch")
            return
    except Exception as e:
        print(f"⛔ Connection failed: {e}")
        return

    while True:
        indices = get_available_indices()
        if not indices:
            print("No indices found or connection failed")
            break

        selected_index, mapping_index = display_index_menu(indices)

        if mapping_index:
            mapping = get_index_mapping(mapping_index)
            if mapping:
                print(f"\nMapping for {mapping_index}:")
                print(json.dumps(mapping, indent=2)[:2000] + "...")
            continue

        if not selected_index:
            break

        print(f"\nAnalyzing index: {selected_index}")
        logs, timestamp_field = get_logs_from_index(selected_index, hours=None)
        if not logs:
            continue

        analyze_with_ollama_chunked(logs, timestamp_field, selected_index)

if __name__ == "__main__":
    main()
