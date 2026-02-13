import tinker
import urllib.request

sc = tinker.ServiceClient()
rc = sc.create_rest_client()

future = rc.get_checkpoint_archive_url_from_tinker_path(
    "tinker://5f0124b7-5ea1-56f6-ac5a-cc3d1173da0c:train:0/sampler_weights/000030"
)
checkpoint_archive_url_response = future.result()

# Download the archive (signed URL, valid until checkpoint_archive_url_response.expires)
urllib.request.urlretrieve(checkpoint_archive_url_response.url, "archive.tar")