import requests, json

session_id = 'test123'
print('=== Turn 1 ===')
r1 = requests.post('http://127.0.0.1:8000/ask', json={'question': 'What is the leave policy?', 'session_id': session_id}, timeout=90)
d1 = r1.json()
print(f'Answer: {d1["answer"][:200]}...')
print(f'Session ID returned: {d1.get("session_id")}')

print('\n=== Turn 2 (follow-up) ===')
r2 = requests.post('http://127.0.0.1:8000/ask', json={'question': 'Can you elaborate on the parental leave part?', 'session_id': session_id}, timeout=90)
d2 = r2.json()
print(f'Answer: {d2["answer"][:300]}...')

print('\n=== Turn 3 (context switch) ===')
r3 = requests.post('http://127.0.0.1:8000/ask', json={'question': 'What about sick leave?', 'session_id': session_id}, timeout=90)
d3 = r3.json()
print(f'Answer: {d3["answer"][:300]}...')