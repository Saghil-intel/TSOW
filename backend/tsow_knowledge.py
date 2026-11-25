
# ============================================================
# TSOW FIELD KNOWLEDGE
# Generic field definitions for questionnaire extraction.
#
# IMPORTANT:
# - Do not add questionnaire-specific expected answers here.
# - Do not use questionnaire_hints to select source rows.
# - The complete questionnaire is provided to the LLM.
# - These definitions explain what information belongs in
#   each TSOW field.
# ============================================================

TSOW_FIELD_KNOWLEDGE = [

    {
        "standard_field": "Company Name",
        "output_path": "company_name",
        "description": (
            "The official name of the company, organization, "
            "ISV, or business participating in the engagement."
        ),
        "alternate_names": [
            "Company", "Organization", "Organization Name",
            "ISV Name", "Partner Name", "Business Name"
        ],
        "matching_guidance": (
            "Identify the organization participating in the engagement. "
            "Distinguish the company from its product, platform, "
            "customers, and technology partners. Search the entire "
            "questionnaire because the company name may appear in "
            "any answer."
        ),
        "answer_format": (
            "Preserve the official organization name. Include a "
            "commonly stated abbreviation when useful."
        )
    },

    {
        "standard_field": "Product Name",
        "output_path": "product_name",
        "description": (
            "The name of the product, software, platform, application, "
            "service, or solution being developed, evaluated, or optimized."
        ),
        "alternate_names": [
            "Product", "Solution Name", "Application Name",
            "Platform Name", "Software Product", "Product / Solution"
        ],
        "matching_guidance": (
            "Identify the actual product or solution. Do not confuse "
            "the product with the company, an individual AI model, "
            "a framework, or a hardware platform. Search all answers "
            "for product references."
        ),
        "answer_format": (
            "Product or solution name, preserving its stated full name "
            "and abbreviation where available."
        )
    },

    {
        "standard_field": "BU",
        "output_path": "bu",
        "description": (
            "The business unit, organizational division, or formal "
            "business group associated with the engagement."
        ),
        "alternate_names": [
            "Business Unit", "Division", "Business Division",
            "Organization Unit", "BU"
        ],
        "matching_guidance": (
            "Extract only an explicitly identified organizational "
            "business unit or division. Do not infer a BU from a "
            "product category, customer industry, cloud provider, "
            "hardware platform, or technical workload."
        ),
        "answer_format": "Business unit or division name."
    },

    {
        "standard_field": "Key Contacts",
        "output_path": "key_contacts",
        "description": (
            "People involved in the engagement, including their names, "
            "roles, responsibilities, and organizational affiliations."
        ),
        "alternate_names": [
            "Contacts", "Key Stakeholders", "Project Contacts",
            "Stakeholders", "Engagement Team"
        ],
        "matching_guidance": (
            "Identify explicitly named people and their roles. Preserve "
            "the distinction between Intel, ISV, and CRM contacts. "
            "Do not infer a person's identity from an email domain, "
            "company name, or job function."
        ),
        "answer_format": (
            "Place supported contact information in the appropriate "
            "Intel, ISV, or CRM subfield. Use null when unavailable."
        )
    },

    {
        "standard_field": "Value Proposition",
        "output_path": "value_proposition",
        "description": (
            "The business and technical value of the solution, including "
            "the problem it addresses, benefits to users or customers, "
            "differentiators, and expected business impact."
        ),
        "alternate_names": [
            "Business Value", "Solution Benefits", "Customer Value",
            "Value Statement", "Business Impact"
        ],
        "matching_guidance": (
            "Extract the complete explanation of why the solution "
            "matters and what value it provides. Look beyond sections "
            "labelled product details. Include relevant capabilities, "
            "problems addressed, customer benefits, and stated business "
            "objectives. Do not reduce a detailed explanation to a "
            "generic product description."
        ),
        "answer_format": (
            "A detailed, readable description preserving all important "
            "business benefits and technical value."
        )
    },

    {
        "standard_field": "Targeted AI Approach",
        "output_path": "targeted_ai_approach",
        "description": (
            "The overall AI methodology, architecture, models, "
            "workloads, software technologies, optimization methods, "
            "operating system, and target hardware used or planned "
            "for the solution."
        ),
        "alternate_names": [
            "AI Approach", "AI Strategy", "AI Architecture",
            "AI Methodology", "Target AI Architecture"
        ],
        "matching_guidance": (
            "This is a parent section. Populate its existing child "
            "fields rather than creating a separate value. Classify "
            "each technical detail according to the definitions of "
            "the child fields. A single questionnaire answer may "
            "provide information for multiple child fields."
        ),
        "answer_format": (
            "Populate the existing nested fields individually."
        )
    },

    {
        "standard_field": "Use Case",
        "output_path": "targeted_ai_approach.use_case",
        "description": (
            "The real-world business or technical scenario in which "
            "the AI solution is applied, including the task it performs "
            "and the users or processes it serves."
        ),
        "alternate_names": [
            "Application", "Business Use Case", "AI Use Case",
            "Usage Scenario", "Target Application"
        ],
        "matching_guidance": (
            "Identify the actual application of the solution. Look "
            "throughout the questionnaire for user-facing scenarios, "
            "business problems, and intended outcomes. Do not confuse "
            "a use case with a model name or implementation framework."
        ),
        "answer_format": (
            "Describe the supported use cases, preserving meaningful "
            "details rather than returning only a broad category."
        )
    },

    {
        "standard_field": "AI Workload",
        "output_path": "targeted_ai_approach.ai_workload",
        "description": (
            "The AI and data-processing activities performed by the "
            "solution, including training, inference, fine-tuning, "
            "classification, generation, streaming, graph processing, "
            "concurrency, and other computational workloads."
        ),
        "alternate_names": [
            "Workload", "AI Processing Requirements",
            "Training Workload", "Inference Workload",
            "Computational Requirements"
        ],
        "matching_guidance": (
            "Identify what computation the system performs, how it "
            "processes data, and any stated scale or concurrency. "
            "Distinguish workload descriptions from hardware "
            "specifications and measured performance results. "
            "Preserve model names and technologies when they explain "
            "the workload."
        ),
        "answer_format": (
            "A detailed description of workload types, processing "
            "activities, scale, and concurrency where stated."
        )
    },

    {
        "standard_field": "Intel Software Stack",
        "output_path": "targeted_ai_approach.intel_software_stack",
        "description": (
            "Intel software products and technologies used or targeted "
            "for the solution, together with relevant existing frameworks "
            "when needed to explain the software environment or migration."
        ),
        "alternate_names": [
            "Software Stack", "Technology Stack", "Intel Software",
            "Software Frameworks", "Development Framework"
        ],
        "matching_guidance": (
            "Identify Intel software technologies explicitly mentioned, "
            "including libraries, SDKs, runtimes, and optimization tools. "
            "Preserve the distinction between currently used software "
            "and proposed or target Intel technologies. Include relevant "
            "non-Intel frameworks only when needed to explain the current "
            "stack, integration, or migration path. Do not classify "
            "hardware or operating systems here."
        ),
        "answer_format": (
            "List technologies with their current or target status "
            "and stated purpose where available."
        )
    },

    {
        "standard_field": "Model/s",
        "output_path": "targeted_ai_approach.models",
        "description": (
            "Machine learning, deep learning, generative AI, foundation "
            "models, and other model architectures used or planned "
            "for the solution."
        ),
        "alternate_names": [
            "Models", "AI Models", "ML Models",
            "Foundation Models", "LLMs", "Model Architecture"
        ],
        "matching_guidance": (
            "Extract model names and relevant architectural details. "
            "Preserve model variants, parameter counts, layers, "
            "components, model roles, and the tasks they perform "
            "when stated. Distinguish models from frameworks, "
            "databases, and optimization techniques."
        ),
        "answer_format": (
            "Include model names and their relevant technical "
            "characteristics and roles."
        )
    },

    {
        "standard_field": "Model Optimizations",
        "output_path": "targeted_ai_approach.model_optimizations",
        "description": (
            "Techniques and technologies used or planned to improve "
            "model inference, training efficiency, latency, throughput, "
            "memory consumption, accuracy, or computational efficiency."
        ),
        "alternate_names": [
            "Model Optimization", "Optimization Techniques",
            "AI Optimization", "Inference Optimization",
            "Model Acceleration"
        ],
        "matching_guidance": (
            "Identify explicitly stated optimization methods such as "
            "fine-tuning approaches, parameter-efficient techniques, "
            "quantization, pruning, precision changes, graph optimization, "
            "and hardware acceleration methods. Preserve whether a "
            "technique is currently used, being evaluated, or planned. "
            "Do not invent optimizations merely because they are common."
        ),
        "answer_format": (
            "Describe each supported optimization technique and "
            "its stated purpose or status."
        )
    },

    {
        "standard_field": "OS",
        "output_path": "targeted_ai_approach.os",
        "description": (
            "Operating systems and versions used or planned for "
            "development, training, testing, or deployment."
        ),
        "alternate_names": [
            "Operating System", "Platform OS",
            "Deployment OS", "System Software"
        ],
        "matching_guidance": (
            "Extract operating system names, distributions, versions, "
            "and deployment options. Distinguish current environments "
            "from optional or planned environments. Do not confuse "
            "an operating system with a programming language or "
            "software framework."
        ),
        "answer_format": (
            "Preserve operating system names, versions, and stated "
            "deployment context."
        )
    },

    {
        "standard_field": "Intel Target Hardware",
        "output_path": "targeted_ai_approach.intel_target_hardware",
        "description": (
            "Intel hardware platforms and architectures targeted for "
            "development, training, inference, optimization, testing, "
            "or deployment."
        ),
        "alternate_names": [
            "Target Hardware", "Hardware Platform", "Target Platform",
            "Processor", "CPU", "GPU", "Intel Architecture",
            "Deployment Hardware", "Target Processor"
        ],
        "matching_guidance": (
            "Search the entire questionnaire for Intel hardware "
            "references, regardless of the question heading. Identify "
            "Intel processor families, architecture generations, "
            "specific processor models, accelerators, and other Intel "
            "platforms. Include the stated purpose, such as inference "
            "or deployment, when available. Distinguish Intel hardware "
            "from third-party cloud instances and non-Intel hardware. "
            "Do not infer an Intel SKU from an architecture family."
        ),
        "answer_format": (
            "Include each explicitly stated Intel hardware platform, "
            "architecture, model, and its stated purpose."
        )
    },

    {
        "standard_field": "Performance Metrics",
        "output_path": "performance_metrics",
        "description": (
            "Quantitative performance measurements, benchmark results, "
            "performance targets, and workload capacity specifications "
            "used to evaluate the solution."
        ),
        "alternate_names": [
            "Performance", "KPIs", "Benchmark Results",
            "Performance Indicators", "System Metrics",
            "Latency", "Throughput", "Concurrency"
        ],
        "matching_guidance": (
            "Search the complete questionnaire for quantitative or "
            "qualitative performance information, even when it appears "
            "under product details, workload, infrastructure, or other "
            "sections. Look for latency, response time, throughput, "
            "requests per second, tokens or sequences per second, "
            "concurrency, accuracy, FPS, memory utilization, and "
            "other measurable indicators. Clearly distinguish measured "
            "results, desired targets, and workload specifications. "
            "A concurrency figure is not automatically a measured "
            "throughput result. Never invent numerical values."
        ),
        "answer_format": (
            "Preserve each metric, its value and unit, and whether it "
            "is a measured result, target, or workload specification."
        )
    },

    {
        "standard_field": "Success Criteria",
        "output_path": "success_criteria",
        "description": (
            "The business, technical, operational, and measurable "
            "outcomes that define whether the engagement or solution "
            "has achieved its intended objectives."
        ),
        "alternate_names": [
            "Success Metrics", "Acceptance Criteria",
            "Expected Outcomes", "Project Goals",
            "Business Objectives"
        ],
        "matching_guidance": (
            "Search the entire questionnaire for explicit objectives "
            "and conditions for success, including optimization goals, "
            "cost reduction, latency improvement, throughput, privacy, "
            "scalability, deployment outcomes, and acceptance conditions. "
            "These may appear in product descriptions or roadmap answers "
            "rather than a section named success criteria. Preserve "
            "numerical targets when explicitly stated. Distinguish "
            "objectives from achieved results and do not turn a general "
            "aspiration into a measurable commitment."
        ),
        "answer_format": (
            "Include all supported success objectives and acceptance "
            "conditions, preserving their scope and measurable targets."
        )
    },

    {
        "standard_field": "Dependencies",
        "output_path": "dependencies",
        "description": (
            "Technical, infrastructure, data, software, hardware, "
            "organizational, and external requirements needed to "
            "implement, integrate, test, or deploy the solution."
        ),
        "alternate_names": [
            "Requirements", "Prerequisites", "Technical Dependencies",
            "Project Dependencies", "Infrastructure Requirements"
        ],
        "matching_guidance": (
            "Identify explicitly stated prerequisites, infrastructure, "
            "data sources and formats, software components, cloud "
            "services, hardware resources, integration requirements, "
            "and organizational dependencies. Include deployment "
            "environments and required engineering resources when "
            "they are stated as requirements. Distinguish dependencies "
            "from technologies that are merely examples or optional "
            "alternatives."
        ),
        "answer_format": (
            "Provide a structured, detailed summary of supported "
            "dependencies, preserving names, versions, and purposes."
        )
    }

]
