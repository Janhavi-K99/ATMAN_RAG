import requests, json

# Test login with demo user
print('=== Login ===')
r = requests.post('http://127.0.0.1:8000/auth/login', json={'email': 'demo@atman.com', 'password': 'demo123'}, timeout=30)
print('Status:', r.status_code)
d = r.json()
print('Token:', d["access_token"][:50] + '...')
print('User:', d["name"], '(', d["email"], ')')
token = d['access_token']

# Test profile
print('\n=== Profile ===')
h = {'Authorization': 'Bearer ' + token}
r2 = requests.get('http://127.0.0.1:8000/auth/me', headers=h, timeout=30)
print('Status:', r2.status_code)
print(json.dumps(r2.json(), indent=2))

# Test create session
print('\n=== New Session ===')
r3 = requests.post('http://127.0.0.1:8000/session/new', headers=h, timeout=30)
print('Status:', r3.status_code)
print(json.dumps(r3.json(), indent=2))
session_id = r3.json()['session_id']

# Test ask question
print('\n=== Ask Question ===')
r4 = requests.post('http://127.0.0.1:8000/ask', json={'question': 'What is the leave policy?', 'session_id': session_id}, headers=h, timeout=90)
print('Status:', r4.status_code)
ans = r4.json()["answer"][:200]
print('Answer:', ans + '...')

# Test session history
print('\n=== Session History ===')
r5 = requests.get('http://127.0.0.1:8000/session/' + session_id + '/history', headers=h, timeout=30)
print('Status:', r5.status_code)
print(json.dumps(r5.json(), indent=2))

# Test list sessions
print('\n=== List Sessions ===')
r6 = requests.get('http://127.0.0.1:8000/sessions', headers=h, timeout=30)
print('Status:', r6.status_code)
print(json.dumps(r6.json(), indent=2))