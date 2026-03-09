# test_neo4j.py — Test your Neo4j connection
# Run this with: python3 test_neo4j.py

from neo4j import GraphDatabase, Auth
from dotenv import load_dotenv
import os

# Load your .env file
load_dotenv()

URI      = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")

# Strip quotes if present
if PASSWORD and PASSWORD.startswith('"'):
    PASSWORD = PASSWORD.strip('"')
if PASSWORD and PASSWORD.startswith("'"):
    PASSWORD = PASSWORD.strip("'")

print("=" * 50)
print("Neo4j Connection Test")
print("=" * 50)

# Check if values are loaded
if not URI or URI == "your_neo4j_uri_here":
    print("❌ NEO4J_URI not set in .env file")
    exit()
if not PASSWORD or PASSWORD == "your_neo4j_password_here":
    print("❌ NEO4J_PASSWORD not set in .env file")
    exit()

print(f"✅ URI loaded:      {URI}")
print(f"✅ Username loaded: {USERNAME}")
print(f"✅ Password loaded: {'*' * len(PASSWORD)}")
print()

try:
    # Connect to Neo4j AuraDB
    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD),
        max_connection_lifetime=30,
        connection_timeout=30
    )
    driver.verify_connectivity()
    print("✅ Neo4j connected successfully!")

    # Create a test node
    with driver.session() as session:
        session.run("CREATE (n:TestDomain {name: 'test-site.com', score: 0.92})")
        print("✅ Test node created in database")

        # Read it back
        result = session.run("MATCH (n:TestDomain) RETURN n.name AS name, n.score AS score")
        for record in result:
            print(f"✅ Node found: {record['name']} | score: {record['score']}")

        # Clean up
        session.run("MATCH (n:TestDomain) DELETE n")
        print("✅ Test node cleaned up")

    driver.close()
    print()
    print("=" * 50)
    print("🎉 Neo4j test complete — everything works!")
    print("=" * 50)

except Exception as e:
    print(f"❌ Connection failed: {e}")
    print()
    print("Common fixes:")
    print("1. Check NEO4J_URI in your .env file")
    print("2. Check NEO4J_PASSWORD in your .env file")
    print("3. Make sure your AuraDB instance shows RUNNING on console.neo4j.io")