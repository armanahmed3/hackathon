import requests
import json

# Test creating a class
print("Testing class creation...")
try:
    response = requests.post('http://localhost:8001/api/classes', 
                           json={'class_name': 'TestClass'})
    print(f"Create class response: {response.status_code}")
    print(f"Response body: {response.json()}")
except Exception as e:
    print(f"Error creating class: {e}")

# Test listing classes
print("\nTesting class listing...")
try:
    response = requests.get('http://localhost:8001/api/classes')
    print(f"List classes response: {response.status_code}")
    print(f"Response body: {response.json()}")
except Exception as e:
    print(f"Error listing classes: {e}")

# Test deleting a class
print("\nTesting class deletion...")
try:
    response = requests.delete('http://localhost:8001/api/classes/TestClass')
    print(f"Delete class response: {response.status_code}")
    print(f"Response body: {response.json()}")
except Exception as e:
    print(f"Error deleting class: {e}")