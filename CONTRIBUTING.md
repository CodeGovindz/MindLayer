# Contributing to MindLayer

Thank you for your interest in contributing to MindLayer! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- At least one LLM provider API key (OpenAI, Google, Anthropic, or Groq)

### Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/MindLayer.git
   cd MindLayer
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Run tests**
   ```bash
   pytest
   ```

## 🛠️ Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep functions focused and small

### Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for good test coverage
- Use pytest for testing

### Commit Messages

Use conventional commit format:
```
type(scope): description

feat(clients): add support for new LLM provider
fix(storage): resolve database connection issue
docs(readme): update installation instructions
test(integration): add end-to-end tests
```

## 📝 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   pytest
   python -m universal_memory_layer.cli.main  # Test CLI
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Go to GitHub and create a PR
   - Fill out the PR template
   - Link any related issues

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment**: OS, Python version, package versions
- **Steps to reproduce**: Clear steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Error messages**: Full error messages and stack traces
- **Configuration**: Relevant configuration (without API keys)

## 💡 Feature Requests

For feature requests, please include:

- **Problem**: What problem does this solve?
- **Solution**: Describe your proposed solution
- **Alternatives**: Any alternative solutions considered
- **Use cases**: How would this feature be used?

## 🏗️ Architecture Guidelines

### Adding New LLM Providers

1. Create a new client in `universal_memory_layer/clients/`
2. Inherit from `BaseLLMClient`
3. Implement required methods
4. Add configuration support
5. Update conversation manager
6. Add tests

### Adding New Features

1. Follow the existing architecture patterns
2. Use dependency injection where appropriate
3. Handle errors gracefully
4. Add comprehensive logging
5. Update documentation

## 📚 Documentation

- Update README.md for user-facing changes
- Update API documentation in `docs/`
- Add docstrings to new functions/classes
- Include usage examples

## 🔒 Security

- Never commit API keys or sensitive data
- Use environment variables for configuration
- Follow security best practices
- Report security issues privately

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MindLayer! 🧠✨
