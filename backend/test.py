import validators

def is_valid_url(url: str) -> bool:
    return validators.url(url)

print(is_valid_url('https://rag-20mar.s3.us-east-2.amazonaws.com/results/mistral/Ankit_Deopurkar_Data_Engineer/content.md'))