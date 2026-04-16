import os
import firebase_admin
from firebase_admin import credentials, storage
from datetime import datetime
import random
from dotenv import load_dotenv

load_dotenv()

class FirebaseStorageCRUD:
    def __init__(self):
        self._init_firebase()

    def _init_firebase(self):
        # Prevent double initialization of Firebase Admin app
        if not firebase_admin._apps:
            cred_path = os.environ.get("FIREBASE_CREDENTIALS")
            bucket_name = os.environ.get("FIREBASE_BUCKET")
            
            if cred_path and bucket_name:
                try:
                    import json
                    
                    # If it's a JSON string (used in Render), load it as dictionary
                    if cred_path.strip().startswith("{"):
                        cred_dict = json.loads(cred_path)
                        cred = credentials.Certificate(cred_dict)
                    else:
                        # Otherwise treat it as a file path
                        cred = credentials.Certificate(cred_path)

                    firebase_admin.initialize_app(cred, {
                        'storageBucket': bucket_name
                    })
                except Exception as e:
                    print(f"Error initializing Firebase: {e}")
            else:
                print("Firebase credentials or bucket name not found in environment variables.")

    def _generate_unique_filename(self, extension="jpeg") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"img_{timestamp}_{random.randint(0, 1000000)}.{extension}"

    def upload_image(self, image_bytes: bytes, extension="jpeg", content_type="image/jpeg", folder="gen_images") -> str:
        """Uploads an image file to Firebase Storage and returns the public URL."""
        if not firebase_admin._apps:
            print("Firebase is not initialized. Cannot upload.")
            return None
            
        try:
            bucket = storage.bucket()
            file_name = f"{folder}/{self._generate_unique_filename(extension)}"
            blob = bucket.blob(file_name)
            
            # Upload from bytes
            blob.upload_from_string(image_bytes, content_type=content_type)
            
            # Make the file public so it can be accessed via URL
            blob.make_public()
            
            # Return the public URL
            return blob.public_url
        except Exception as e:
            print(f"Error uploading to Firebase Storage: {e}")
            return None
