## Project Overview
This project is a Python-based application designed for an AI chatbot assistant without relying on external agentic AI frameworks.

### Golden Rules
* When unsure about implementation details or requirements, ALWAYS consult the developer rather than making assumptions.
* Always ask for confirmation before implementing a solution that involves creating new files or deleting existing ones.
* Limit your responses to the specific ask of the developer.

### Coding Standards  
* Follow this philosophy when writing code: "Simplicity is the ultimate sophistication."  
* Functions/methods: Always include a docstring briefly describing its purpose and parameters, along with type hints for parameters and return values. No need to over-explain obvious functionality.  
* Naming: `snake_case` (functions/variables), `PascalCase` (classes), `SCREAMING_SNAKE` (constants).  
* Use the typing library for type hints in both functions and regular objects, with imports such as `List`, `Dict`, `Optional`, `Union`, etc.  
* Always use **logger** from `loguru` for logging instead of generic `print` statements.  
* Limit line length to 100 characters tops, and respect standard PEP8 linting rules.
