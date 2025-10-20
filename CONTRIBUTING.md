# Contributing to PersianAI Blog Generator

Thank you for your interest in contributing to PersianAI Blog Generator! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### **Ways to Contribute**
- 🐛 **Bug Reports**: Report bugs and issues
- ✨ **Feature Requests**: Suggest new features and improvements
- 📝 **Documentation**: Improve documentation and examples
- 🔧 **Code Contributions**: Submit code improvements and new features
- 🧪 **Testing**: Help test the system and report issues
- 🌍 **Translations**: Help translate documentation to other languages

## 🚀 Getting Started

### **Prerequisites**
- Python 3.8+
- Git
- OpenAI API key (for testing)
- Basic understanding of AI/ML concepts

### **Development Setup**

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/persianai-blog-generator.git
   cd persianai-blog-generator
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   ```bash
   cp env.example .env
   # Edit .env and add your OpenAI API key
   ```

4. **Create Development Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Contribution Guidelines

### **Code Style**
- Follow **PEP 8** Python style guidelines
- Use **type hints** for all function parameters and return values
- Write **docstrings** for all functions and classes
- Use **meaningful variable names**
- Keep **functions small and focused**

### **Commit Messages**
Use clear, descriptive commit messages:
```bash
# Good
git commit -m "feat: add advanced RAG content selection algorithm"
git commit -m "fix: resolve Persian typography spacing issues"
git commit -m "docs: update README with scalability information"

# Bad
git commit -m "fix"
git commit -m "update"
git commit -m "changes"
```

### **Pull Request Process**

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write your code
   - Add tests if applicable
   - Update documentation
   - Test your changes

3. **Test Your Changes**
   ```bash
   # Run basic tests
   python generate_blog.py --keyword "تست" --out outputs/test.html
   
   # Check for linting errors
   python -m flake8 generate_blog.py
   ```

4. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: your feature description"
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Go to GitHub and create a pull request
   - Fill out the PR template
   - Link any related issues
   - Request review from maintainers

## 🧪 Testing Guidelines

### **Testing Your Changes**
- Test with different keywords and scenarios
- Verify Persian typography rules are followed
- Check content quality metrics
- Ensure RAG content selection works correctly

### **Test Cases to Consider**
- Different keyword types (technical, business, medical)
- Various content lengths and structures
- Edge cases and error conditions
- Performance with large RAG datasets

## 📝 Documentation Guidelines

### **Code Documentation**
- Write clear docstrings for all functions
- Include parameter descriptions and return values
- Add usage examples where helpful
- Document any complex algorithms or logic

### **README Updates**
- Update installation instructions if needed
- Add new features to the features list
- Update usage examples
- Include any new configuration options

## 🐛 Bug Reports

### **Before Reporting**
- Check if the issue already exists
- Try the latest version
- Test with different keywords/scenarios
- Check your environment setup

### **Bug Report Template**
```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Run command '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 10, macOS 12, Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- Package versions: [e.g., openai==1.3.0]

**Additional Context**
Any other relevant information.
```

## ✨ Feature Requests

### **Before Requesting**
- Check if the feature already exists
- Consider if it fits the project's scope
- Think about implementation complexity
- Consider backward compatibility

### **Feature Request Template**
```markdown
**Feature Description**
A clear description of the feature.

**Use Case**
Why would this feature be useful?

**Proposed Solution**
How would you like to see this implemented?

**Alternatives**
Any alternative solutions you've considered.

**Additional Context**
Any other relevant information.
```

## 🔧 Development Areas

### **High Priority**
- Performance optimizations
- Additional Persian language rules
- Better error handling and logging
- More comprehensive testing
- Documentation improvements

### **Medium Priority**
- Multi-language support
- Advanced analytics
- Custom model fine-tuning
- API endpoint development
- UI/UX improvements

### **Low Priority**
- Voice-to-blog generation
- Video content generation
- Advanced AI model integration
- Enterprise features

## 📊 Code Review Process

### **Review Criteria**
- **Functionality**: Does the code work as intended?
- **Performance**: Is it efficient and scalable?
- **Security**: Are there any security concerns?
- **Maintainability**: Is the code easy to understand and maintain?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the code well-documented?

### **Review Process**
1. **Automated Checks**: CI/CD runs automated tests
2. **Code Review**: Maintainers review the code
3. **Testing**: Manual testing of new features
4. **Approval**: Code is approved and merged

## 🏷️ Release Process

### **Version Numbering**
We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### **Release Steps**
1. Update version numbers
2. Update CHANGELOG.md
3. Create release notes
4. Tag the release
5. Publish to PyPI (if applicable)

## 🤔 Questions?

### **Getting Help**
- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For general questions and discussions
- **Email**: Contact maintainers directly
- **Documentation**: Check the README and code comments

### **Community Guidelines**
- **Be Respectful**: Treat everyone with respect
- **Be Constructive**: Provide helpful feedback
- **Be Patient**: Remember this is a volunteer project
- **Be Collaborative**: Work together to improve the project

## 🙏 Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Mentioned in release announcements
- **GitHub**: Listed as contributors on the repository

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to PersianAI Blog Generator!** 🚀

Your contributions help make this project better for everyone in the Persian content creation community.
