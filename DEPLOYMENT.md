# Simple EC2 Docker deployment

This deployment uses one Ubuntu EC2 instance, its `.pem` SSH key, Docker
Compose, and ordinary AWS access-key credentials. Every push to `main` runs the
tests and replaces the running API container.

The public endpoints will be:

```text
http://<EC2_PUBLIC_IP>:8010/docs
http://<EC2_PUBLIC_IP>:8010/health
```

## 1. Create the EC2 instance

1. Launch an Ubuntu 24.04 LTS EC2 instance.
2. Create/download an RSA `.pem` key pair.
3. Paste `infra/ec2-user-data.sh` into **Advanced details > User data**. This
   installs Docker Engine and Docker Compose.
4. Assign an Elastic IP if the API address must remain unchanged after an
   instance stop/start.
5. Wait for the instance status checks and user-data installation to finish.

## 2. Security-group inbound ports

Add these inbound rules:

| Port | Source | Purpose |
| --- | --- | --- |
| TCP `22` | `0.0.0.0/0` for the simplest GitHub-hosted deployment | GitHub Actions SSH deployment |
| TCP `8010` | `0.0.0.0/0`, or preferably your backend server IP `/32` | Public API |

GitHub-hosted runner IPs change, so restricting port 22 only to your home IP
will prevent GitHub Actions from connecting. SSH remains key-only; never enable
password login.

## 3. Bedrock IAM user

Create an IAM user with programmatic access and attach the policy in
`infra/bedrock-iam-policy.json`. Create one access key and copy its access-key
ID and secret access key.

The AWS account must already be verified and permitted to use
`zai.glm-4.7-flash` in `eu-west-1`.

## 4. GitHub production configuration

Open **Repository > Settings > Environments** and create `production`.

Add these environment secrets:

| Secret | Value |
| --- | --- |
| `AWS_ACCESS_KEY_ID` | Bedrock IAM user's access-key ID |
| `AWS_SECRET_ACCESS_KEY` | Bedrock IAM user's secret access key |
| `EC2_SSH_PRIVATE_KEY` | The complete `.pem` file, including BEGIN/END lines |
| `EC2_HOST` | EC2 public/Elastic IP without `http://` |

## 5. Deploy

Push or merge into `main`. The workflow at `.github/workflows/ci-cd.yml` will:

1. Run the Python tests and build the Docker image.
2. Connect to EC2 using the PEM key.
3. copy the application and a permission-restricted production environment
   file.
4. Run `docker compose up`.
5. Verify the API from both EC2 and the public internet.

The deployment URL is printed in the GitHub Actions job summary. You can also
manually rerun the workflow from the Actions tab after rotating a secret.

This direct-IP setup uses HTTP. Add a domain and an HTTPS reverse proxy later
if browsers or external customers will call the service directly.
