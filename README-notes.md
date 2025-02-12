bq show mdepew-assets:synthetic.synthetic_mortgage_data


export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/app
lication_default_credentials.json


base64 ~/.config/gcloud/application_default_credentials.json > credentials_base64.txt && CREDS=$(cat credentials_base64.txt) && sed "s/\${BASE64_ENCODED_CREDENTIALS}/$CREDS/" manifests/bigquery-secret.yaml > manifests/bigquery-secret-updated.yaml