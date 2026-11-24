# questionnaire_hints:
#   list of lowercase phrases. A questionnaire row is used for this field only
#   if its QUESTION (first cell) contains one of these phrases.
#   [] means the questionnaire has no source for this field (skip the LLM).
#   Omit the key entirely to fall back to semantic retrieval.
#
# answer_format:
#   short instruction telling the LLM what shape the answer must have.

TSOW_FIELD_KNOWLEDGE = [

    {
        "standard_field": "Company Name",
        "description": "Official name of the organization or company participating in the engagement.",
        "alternate_names": ["Company", "Organization", "Organization Name", "ISV Name", "Partner Name", "Business Name"],
        "examples": ["Assessli", "Microsoft", "Intel partner organization"],
        "matching_guidance": "Identify the company or organization. Do not confuse it with the product name.",
        "questionnaire_hints": ["product details"],
        "answer_format": "Company name only, e.g. Assessli"
    },
    {
        "standard_field": "Product Name",
        "description": "Name of the product, software, platform, application or service being developed or evaluated.",
        "alternate_names": ["Product", "Solution Name", "Application Name", "Platform Name", "Software Product", "Product / Solution"],
        "examples": ["Large Behavioral Model", "AI recommendation platform"],
        "matching_guidance": "Identify the actual product or solution, not just an individual AI model.",
        "questionnaire_hints": ["product details"],
        "answer_format": "Product name only, e.g. Large Behavioral Model (LBM)"
    },
    {
        "standard_field": "BU",
        "description": "Business Unit or organizational division associated with the company, product or engagement.",
        "alternate_names": ["Business Unit", "Division", "Business Division", "Organization Unit"],
        "examples": ["Data Center Group", "Client Computing Group"],
        "matching_guidance": "Identify an organizational business division, not hardware or software.",
        "questionnaire_hints": [],
        "answer_format": "Business unit name only"
    },
    {
        "standard_field": "Key Contacts",
        "description": "Names, roles and contact details of stakeholders involved in the engagement.",
        "alternate_names": ["Contacts", "Key Stakeholders", "Project Contacts", "Stakeholders", "Engagement Team"],
        "examples": ["Intel contact", "ISV contact", "CRM contact", "Technical lead"],
        "matching_guidance": "Identify people, roles and organizational affiliations. Preserve contact subcategories such as Intel, ISV and CRM.",
        "questionnaire_hints": [],
        "answer_format": "Names with roles, comma-separated"
    },
    {
        "standard_field": "Value Proposition",
        "description": "Business value, benefits, competitive advantages and problems addressed by the product or solution.",
        "alternate_names": ["Business Value", "Solution Benefits", "Customer Value", "Value Statement", "Business Impact"],
        "examples": ["Reduced inference costs", "Improved customer experience"],
        "matching_guidance": "Identify why the solution is valuable and what benefits it provides. Do not restate the product description.",
        "questionnaire_hints": ["product details"],
        "answer_format": "One short sentence on the benefit to the customer or Intel"
    },
    {
        "standard_field": "Targeted AI Approach",
        "description": "AI methodology, architecture and technical approach used or planned for implementing the solution.",
        "alternate_names": ["AI Approach", "AI Strategy", "AI Architecture", "AI Methodology", "Target AI Architecture"],
        "examples": ["Retrieval-Augmented Generation", "Agentic AI", "Graph-based reasoning"],
        "matching_guidance": "Identify the overall AI methodology or architecture, not just a model name.",
        "questionnaire_hints": ["product details", "topologies"],
        "answer_format": "One short sentence naming the AI methodology"
    },
    {
        "standard_field": "Use Case",
        "description": "Specific business scenario or practical application where the product or AI solution is used.",
        "alternate_names": ["Application", "Business Use Case", "AI Use Case", "Usage Scenario", "Target Application"],
        "examples": ["Personalized recommendations", "Predictive maintenance", "Customer behavior analysis"],
        "matching_guidance": "Identify what the solution does in a real-world scenario.",
        "questionnaire_hints": ["product details", "workload details"],
        "answer_format": "One short phrase naming the application"
    },
    {
        "standard_field": "AI Workload",
        "description": "Nature, scale and computational requirements of AI workloads, including training, inference, data volume and concurrency.",
        "alternate_names": ["Workload", "AI Processing Requirements", "Training Workload", "Inference Workload", "Computational Requirements"],
        "examples": ["Real-time inference", "Large-scale model training", "Streaming data processing"],
        "matching_guidance": "Identify the actual AI processing workload, not merely the hardware.",
        "questionnaire_hints": ["workload details"],
        "answer_format": "Short semicolon-separated list of workload types, with concurrency if stated"
    },
    {
        "standard_field": "Intel Software Stack",
        "description": "Intel software tools, frameworks, libraries, SDKs and optimization technologies used or planned.",
        "alternate_names": ["Software Stack", "Technology Stack", "Intel Software", "Software Frameworks", "Development Framework"],
        "examples": ["OpenVINO", "oneAPI", "Intel oneDNN"],
        "matching_guidance": "Identify software technologies and frameworks. Do not classify hardware or operating systems here.",
        "questionnaire_hints": ["frameworks"],
        "answer_format": "Comma-separated Intel software names only (e.g. OpenVINO, IPEX); exclude third-party tools"
    },
    {
        "standard_field": "Model/s",
        "description": "Names and types of machine learning, deep learning, generative AI or other models used or planned.",
        "alternate_names": ["Models", "AI Models", "ML Models", "Foundation Models", "LLMs", "Model Architecture"],
        "examples": ["Llama", "Qwen", "BERT", "Large Behavioral Model"],
        "matching_guidance": "Identify actual model names and types. Distinguish models from optimization techniques.",
        "questionnaire_hints": ["topologies"],
        "answer_format": "Comma-separated model names only"
    },
    {
        "standard_field": "Model Optimizations",
        "description": "Techniques applied or planned to improve model performance, efficiency, memory consumption or inference speed.",
        "alternate_names": ["Model Optimization", "Optimization Techniques", "AI Optimization", "Inference Optimization", "Model Acceleration"],
        "examples": ["Quantization", "Pruning", "Knowledge distillation", "Graph optimization"],
        "matching_guidance": "Identify optimization methods, not model names or hardware platforms.",
        "questionnaire_hints": ["workload details", "frameworks"],
        "answer_format": "Comma-separated optimization techniques only"
    },
    {
        "standard_field": "OS",
        "description": "Operating system and version used or planned for development, training or deployment.",
        "alternate_names": ["Operating System", "Platform OS", "Deployment OS", "System Software"],
        "examples": ["Ubuntu Linux", "Red Hat Enterprise Linux", "Windows Server"],
        "matching_guidance": "Identify the operating system, not the software framework.",
        "questionnaire_hints": ["what os"],
        "answer_format": "OS name and version only, e.g. Ubuntu 24.04 LTS"
    },
    {
        "standard_field": "Intel Target Hardware",
        "description": "Intel hardware platforms targeted for development, training, optimization or inference.",
        "alternate_names": ["Target Hardware", "Hardware Platform", "Target Platform", "Processor", "CPU", "GPU", "Intel Architecture", "Deployment Hardware"],
        "examples": ["Intel Xeon 6", "Intel Core Ultra", "Intel Gaudi"],
        "matching_guidance": "Identify Intel hardware platforms the solution is being optimized for. Do not list cloud instances or non-Intel hardware.",
        "questionnaire_hints": ["product details"],
        "answer_format": "Intel hardware names only, e.g. Intel Xeon 6"
    },
    {
        "standard_field": "Performance Metrics",
        "description": "Quantitative measurements used to evaluate AI solution or hardware performance.",
        "alternate_names": ["Performance", "KPIs", "Benchmark Results", "Performance Indicators", "System Metrics"],
        "examples": ["Inference latency", "Throughput", "Tokens per second", "Accuracy", "Memory utilization"],
        "matching_guidance": "Identify measurable performance indicators and values. Distinguish targets from measured results. Do not list hardware instances.",
        "questionnaire_hints": ["product details", "workload details"],
        "answer_format": "Comma-separated metric names; include values only if stated"
    },
    {
        "standard_field": "Success Criteria",
        "description": "Expected outcomes, measurable goals and acceptance conditions used to determine engagement success.",
        "alternate_names": ["Success Metrics", "Acceptance Criteria", "Expected Outcomes", "Project Goals", "Business Objectives"],
        "examples": ["Achieve target inference latency", "Improve throughput", "Successfully deploy on Intel hardware"],
        "matching_guidance": "Identify what must be achieved for the project to be considered successful.",
        "questionnaire_hints": [],
        "answer_format": "Short list of measurable goals"
    },
    {
        "standard_field": "Dependencies",
        "description": "External requirements, prerequisites, resources and technical or organizational dependencies required for implementation.",
        "alternate_names": ["Requirements", "Prerequisites", "Technical Dependencies", "Project Dependencies", "Infrastructure Requirements"],
        "examples": ["Hardware availability", "Software compatibility", "Dataset availability", "Engineering resources"],
        "matching_guidance": "Identify resources or conditions required for implementation.",
        "questionnaire_hints": [],
        "answer_format": "Short comma-separated list"
    }

]