import os
import random
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv
from typing import Any
from supabase.lib.client_options import SyncClientOptions

load_dotenv()


class AttrDict(dict):
    """Dictionary subclass that allows attribute access to keys and recursively
    converts nested dicts/lists to AttrDicts for dot-access convenience.
    Keeps dict-like behavior (including `.get()`) for compatibility.
    """
    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @staticmethod
    def convert(obj: Any) -> Any:
        """Recursively converts dicts and lists to AttrDicts."""
        if isinstance(obj, dict):
            return AttrDict({k: AttrDict.convert(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [AttrDict.convert(v) for v in obj]
        return obj

class SupabaseCRUD:
    def __init__(self):
        self.supabase_client = self._init_supabase_client()

    def _init_supabase_client(self) -> Client:
        url = os.environ.get("SUPABASE_URL") 
        key = os.environ.get("SUPABASE_KEY")
        
        options = SyncClientOptions(postgrest_client_timeout=60)
        return create_client(url, key, options=options)

    def _generate_unique_filename(self, extension="png") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"img_{timestamp}_{random.randint(0, 1000000)}.{extension}"

    def _get_public_url(self, bucket_name: str, file_name: str) -> str:
        return self.supabase_client.storage.from_(bucket_name).get_public_url(file_name)

    def upload_image(
        self,
        image_bytes: bytes,
        file_name: str = None,
        bucket_name="demo-bucket",
        content_type="image/png") -> str:
        """Uploads an image file to Supabase storage and returns the public URL."""

        if not file_name:
            file_name = self._generate_unique_filename()

        self.supabase_client.storage.from_(bucket_name).upload(
            path=file_name,
            file=image_bytes,
            file_options={
                "content-type": content_type,
                "upsert": "true"
            }
        )

        return self._get_public_url(bucket_name, file_name)
    
    def upload_file(self, file_bytes: bytes, file_name: str = None, bucket_name="demo-bucket", content_type="image/png") -> str:
        """Alias for upload_image to fix potential bug in usage."""
        return self.upload_image(file_bytes, file_name, bucket_name, content_type)

    def insert_row(self, table_name: str, data: dict):
        """Inserts a row into the specified table."""
        response = self.supabase_client.table(table_name).insert(data).execute()
        if response.data:
            return AttrDict.convert(response.data[0])
        return None

    def update_row(self, table_name: str, data: dict, row_id: int):
        """Updates a row by ID in the specified table."""
        response = self.supabase_client.table(table_name).update(data).eq('id', row_id).execute()
        if response.data:
            return AttrDict.convert(response.data[0])
        return None

    def delete_row(self, table_name: str, row_id: int):
        """Deletes a row by ID from the specified table."""
        self.supabase_client.table(table_name).delete().eq('id', row_id).execute()
        return True

    def get_row_by_id(self, table_name: str, row_id: int):
        """Retrieves a single row by its ID."""
        response = self.supabase_client.table(table_name).select('*').eq('id', row_id).execute()
        if response.data:
            return AttrDict.convert(response.data[0])
        return None

    def get_all_rows(self, table_name: str):
        """Retrieves all rows from the specified table."""
        response = self.supabase_client.table(table_name).select('*').execute()
        return AttrDict.convert(response.data)

