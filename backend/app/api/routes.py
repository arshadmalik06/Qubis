import json
import re
import uuid
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from app.core.vector_store import BISVectorStore
from app.core.llm_agent import BISLLMAgent

router = APIRouter()
vector_store = BISVectorStore()
llm_agent = BISLLMAgent()

# ==========================================
# EXISTING CHAT & SEARCH ROUTES (UNTOUCHED)
# ==========================================
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Receives a query, fetches relevant clauses from ChromaDB, 
    and streams the AI response back to the client.
    """
    try:
        context_chunks = vector_store.search(request.query, top_k=8)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database search failed: {str(e)}")

    # Create a custom generator to intercept the stream and inject the source metadata
    async def event_generator():
        # 1. Send the top source metadata as a custom "source" event
        if context_chunks and len(context_chunks) > 0:
            top_metadata = context_chunks[0].get("metadata", {})
            filename = top_metadata.get("filename")
            page_number = top_metadata.get("page_number")
            
            if filename and page_number:
                yield {
                    "event": "source",
                    "data": json.dumps({
                        "filename": filename,
                        "page_number": page_number
                    })
                }
        
        # 2. Stream the actual LLM text tokens
        async for chunk in llm_agent.generate_stream(request.query, context_chunks):
            yield chunk

    return EventSourceResponse(event_generator())


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8

@router.post("/search")
async def search_documents(request: SearchRequest):
    try:
        results = vector_store.search(request.query, top_k=request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# NEW: COMPLIANCE CHECKLIST MODULE
# ==========================================

# 1. Pydantic Models representing the exact data contract
class ChecklistRequest(BaseModel):
    product: str
    intended_use: str

class ChecklistItem(BaseModel):
    id: str
    category: str
    task: str
    description: str
    reference_clause: str

class ChecklistResponse(BaseModel):
    product_name: str
    applicable_standard: str
    items: List[ChecklistItem]


def extract_json_from_llm_response(text: str) -> dict:
    """Bulletproof JSON extractor to strip markdown fences and conversational hallucinations."""
    # Strip markdown code fences
    cleaned_text = re.sub(r'```(?:json)?', '', text).strip()
    cleaned_text = re.sub(r'```', '', cleaned_text).strip()
    
    # Isolate the core JSON object {} in case the LLM typed something before/after
    start = cleaned_text.find('{')
    end = cleaned_text.rfind('}')
    if start != -1 and end != -1:
        cleaned_text = cleaned_text[start:end+1]
    else:
        raise ValueError(f"No JSON object found in LLM output. Raw: {cleaned_text[:500]}")
        
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output into JSON: {e}. Raw output: {cleaned_text[:500]}")


# The dedicated system prompt for checklist generation — completely separate from the chat agent
CHECKLIST_SYSTEM_PROMPT = """You are an expert BIS (Bureau of Indian Standards) compliance consultant.
Your task is to generate a structured, product-specific compliance checklist in strict JSON format.

You are authoritative on BIS certification schemes (ISI Mark, BIS Registration, CRS, FMCS),
Indian Standards (IS codes), mandatory testing requirements, documentation for license applications,
and general regulatory compliance workflows under the BIS Act 2016 and related rules.

CRITICAL RULES:
1. Output ONLY valid JSON. No greetings, no explanations, no markdown, no commentary.
2. Generate comprehensive, actionable checklist items covering ALL compliance phases.
3. If provided context contains relevant BIS standards, use them. Otherwise, use your expert knowledge.
4. Each checklist item must be specific, actionable, and professionally worded.
5. The JSON must strictly follow the exact schema provided in the prompt."""


@router.post("/checklist/generate", response_model=ChecklistResponse)
async def generate_compliance_checklist(request: ChecklistRequest):
    try:
        # 1. Targeted Vector Search — retrieve any relevant BIS context
        search_query = (
            f"BIS compliance certification testing documentation requirements "
            f"for {request.product} used for {request.intended_use}"
        )
        context_chunks = []
        try:
            context_chunks = vector_store.search(search_query, top_k=6)
        except Exception:
            # If vector store fails, we continue without context — the LLM
            # can still generate a useful checklist from its training knowledge
            pass

        # 2. Format retrieved context for the prompt
        context_section = ""
        if context_chunks:
            context_section = "\n\nRELEVANT BIS REFERENCE MATERIAL:\n"
            for i, chunk in enumerate(context_chunks):
                content = chunk.get("content", "")
                meta = chunk.get("metadata", {})
                std_id = meta.get("standard_id", "Unknown")
                clause_id = meta.get("clause_id", "Unknown")
                context_section += (
                    f"\n--- Reference {i+1} (Source: {std_id}, Clause {clause_id}) ---\n"
                    f"{content}\n"
                )
        else:
            context_section = (
                "\n\nNote: No specific BIS documents were found in the database for this product. "
                "Generate the checklist based on your expert knowledge of BIS certification processes, "
                "applicable Indian Standards, and general regulatory compliance requirements.\n"
            )

        # 3. Strict JSON-only prompt
        prompt = f"""Generate a comprehensive BIS compliance checklist for the following product.

Product Name: {request.product}
Intended Market / Use: {request.intended_use}
{context_section}

You MUST generate items across ALL of these categories:
- "Documentation" — application forms, technical documents, test reports, declarations
- "Testing" — laboratory testing, sample preparation, type tests, routine tests
- "Certification" — BIS license application, scheme selection (ISI/CRS/FMCS), factory inspection
- "Marking & Labeling" — ISI mark usage, product labeling, packaging requirements
- "Post-Certification" — surveillance audits, renewal, record keeping, non-conformity handling

Generate at least 8 checklist items spread across these categories.

You MUST output ONLY valid JSON matching this EXACT structure (no other text):
{{
    "product_name": "{request.product}",
    "applicable_standard": "IS XXXX:YYYY - Standard Title (or 'General BIS Compliance Guidelines' if no specific standard is known)",
    "items": [
        {{
            "category": "Documentation",
            "task": "Concise task name",
            "description": "Detailed, actionable instruction on what needs to be done",
            "reference_clause": "IS standard number and clause, or 'BIS Act 2016' if general"
        }}
    ]
}}

OUTPUT ONLY THE JSON. NO OTHER TEXT."""

        # 4. Call the LLM with non-streaming completion (clean, no SSE parsing)
        raw_response = await llm_agent.generate_completion(
            prompt=prompt,
            system_prompt=CHECKLIST_SYSTEM_PROMPT
        )

        if not raw_response or not raw_response.strip():
            raise ValueError("LLM returned an empty response. Ollama may be overloaded.")

        # 5. Extract and parse JSON from the response
        parsed_json = extract_json_from_llm_response(raw_response)

        # 6. Validate required fields exist
        if "items" not in parsed_json or not isinstance(parsed_json.get("items"), list):
            raise ValueError("LLM response missing 'items' array.")
        
        if len(parsed_json["items"]) == 0:
            raise ValueError("LLM generated an empty checklist.")

        # 7. Ensure product_name and applicable_standard are present
        parsed_json.setdefault("product_name", request.product)
        parsed_json.setdefault("applicable_standard", "General BIS Compliance Guidelines")

        # 8. Inject UUIDs and sanitize each item for the frontend
        sanitized_items = []
        for item in parsed_json["items"]:
            sanitized_items.append({
                "id": str(uuid.uuid4()),
                "category": str(item.get("category", "General")),
                "task": str(item.get("task", "Untitled Task")),
                "description": str(item.get("description", "")),
                "reference_clause": str(item.get("reference_clause", "N/A")),
            })
        parsed_json["items"] = sanitized_items

        return parsed_json
        
    except ConnectionError as ce:
        raise HTTPException(status_code=503, detail=str(ce))
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=f"Checklist parsing failed: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Checklist generation failed: {str(e)}")