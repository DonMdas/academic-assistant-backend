from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

def get_id_token():
    flow = InstalledAppFlow.from_client_secrets_file(
        "credentials.json",
        scopes=SCOPES
    )

    # 👇 Force port 8080
    creds = flow.run_local_server(
        host="localhost",
        port=8080,
        authorization_prompt_message="Opening browser for Google login...",
        success_message="Authentication successful. You can close this tab."
    )

    id_token = creds.id_token

    print("\n=== ID TOKEN ===\n")
    print(id_token)

    return id_token


if __name__ == "__main__":
    get_id_token()