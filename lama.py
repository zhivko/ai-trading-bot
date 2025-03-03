# https://frameworks.readthedocs.io/en/latest/ai/openWebuiDocker.html
# git clone https://github.com/open-webui/open-webui.git
# cd open-webui

# v ollama: section moraš dodati:
#    ports:
#      - 11434:11434/tcp
# če želiš dostopati to ollama containerja

# docker compose -f docker-compose.yaml -f docker-compose.gpu.yaml up -d --build
# docker exec -it ollama ollama run llama3.2
# lama.py

import sys

print(sys.version)

from ollama import Client
import json
import time

i = 0

while i < 1000:
    try:
        print("try no: " + str(i))
        client = Client(
        host='http://192.168.64.116:11434',
        headers={'Content-Type': 'application/json'}
        )
        #response = client.chat(model='llama3.2', messages=[
        response = client.chat(
            model='deepseek-r1:7b', 
            messages=[
                {
                    'role': 'user',
                    'content': 'DO you know origins of name Vid?',
                },
            ],
        )

        print(response)
        json_response = json.dumps(response)

        # Parse the JSON string back into a dictionary
        parsed_response = json.loads(json_response)

        print(parsed_response['message']['content'])

        

    except Exception as x:
        print(x)

    time.sleep(2)

    i = i + 1
