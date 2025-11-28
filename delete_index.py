import asyncio
import yaml
from backend.opensearch_client import OpenSearchClient

async def delete_index():
    # Load config
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Initialize client
    client = OpenSearchClient(config)
    index_name = config['opensearch']['index_name']
    
    try:
        # Check if index exists
        exists = await client.client.indices.exists(index=index_name)
        if exists:
            # Delete index
            await client.client.indices.delete(index=index_name)
            print(f"✅ Successfully deleted index: {index_name}")
        else:
            print(f"⚠️ Index '{index_name}' does not exist.")
            
    except Exception as e:
        print(f"❌ Error deleting index: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(delete_index())