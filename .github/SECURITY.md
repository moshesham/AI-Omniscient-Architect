# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | :white_check_mark: |
| 1.x     | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in Omniscient Architect, please report it by:

1. **Email**: Create a private security advisory on GitHub
2. **Do NOT** create a public issue for security vulnerabilities

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on severity

## Security Best Practices

When using Omniscient Architect:

1. **Local Analysis**: All code analysis runs locally by default
2. **No Data Transmission**: Your code never leaves your machine when using Ollama
3. **API Keys**: Store GitHub tokens securely using environment variables
4. **Docker**: Use the provided Docker configuration for isolated environments
