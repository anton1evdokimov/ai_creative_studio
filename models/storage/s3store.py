"""MinIO / S3 object store for uploaded product photos."""
from __future__ import annotations

import os

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def bucket_name() -> str:
    return os.environ.get("S3_BUCKET", "studio-uploads")


def _client():
    endpoint = (os.environ.get("S3_ENDPOINT") or "").strip()
    if not endpoint:
        raise RuntimeError("S3_ENDPOINT is not set (MinIO)")
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY", "minio"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY", "minioadmin"),
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        config=Config(s3={"addressing_style": "path"}),
    )


def ensure_bucket() -> str:
    name = bucket_name()
    client = _client()
    try:
        client.head_bucket(Bucket=name)
    except Exception:
        client.create_bucket(Bucket=name)
    return name


def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    name = ensure_bucket()
    _client().put_object(Bucket=name, Key=key, Body=data, ContentType=content_type)
    return key


def list_image_keys(prefix: str) -> list[str]:
    name = bucket_name()
    client = _client()
    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": name, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for obj in page.get("Contents") or []:
            key = str(obj.get("Key") or "")
            ext = "." + key.rsplit(".", 1)[-1].lower() if "." in key else ""
            if ext in _IMG_EXT:
                keys.append(key)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    keys.sort()
    return keys


def get_bytes(key: str) -> bytes:
    name = bucket_name()
    body = _client().get_object(Bucket=name, Key=key)["Body"]
    return body.read()
