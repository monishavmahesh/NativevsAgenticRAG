import re
from sympy import symbols, Eq, solve
from core.vectorstore import build_vector_store
from langchain_community.document_loaders import PyPDFLoader

class BaseAgent:
    def __init__(self, subject: str, pdf_path: str, llm, memory):
        self.subject = subject
        self.pdf_path = pdf_path
        self.llm = llm
        self.memory = memory
        
        # Load textbook
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"[BaseAgent] Loaded {len(docs)} pages from {pdf_path}")
        
        if not docs:
            raise ValueError(f"No documents found in {pdf_path}")
        
        self.vectordb = build_vector_store(subject, docs)
    
    def _solve_equation(self, question: str):
        """Detect and solve simple math equations using SymPy."""
        try:
            if self.subject.lower() != "math":
                return None
            
            match = re.search(r"([0-9xX\+\-\*/\s]+)=([0-9\+\-\*/\s]+)", question)
            if not match:
                return None
            
            left_expr, right_expr = match.groups()
            x = symbols('x')
            left_expr = left_expr.replace("X", "x")
            eq = Eq(eval(left_expr.replace("x", "*x")), eval(right_expr))
            solution = solve(eq, x)
            
            if solution:
                return f"Solution: x = {solution}"
            else:
                return "No real solution found."
        except Exception:
            return None
    
    def query(self, question: str) -> str:
        # Save user message into memory
        self.memory.chat_memory.add_user_message(question)
        
        # ✅ FIXED: More flexible subject filter for Math
        q_lower = question.lower()
        
        if self.subject.lower() == "math":
            # Expanded keywords + more flexible matching
            math_keywords = [
                "x", "y", "z", "equation", "solve", "calculate", "value", "algebra", 
                "add", "subtract", "multiply", "divide", "number", "sum", "difference",
                "product", "quotient", "fraction", "decimal", "percentage", "ratio",
                "area", "perimeter", "volume", "square", "cube", "root", "power",
                "how many", "how much", "total", "altogether", "left", "more", "less",
                "times", "divided", "plus", "minus", "equal", "greater", "smaller",
                "digit", "place value", "round", "estimate", "measure", "length",
                "width", "height", "math", "maths", "mathematics", "problem"
            ]
            
            # ✅ FIXED: Allow if ANY keyword matches OR if it's a simple number question
            has_math_keyword = any(word in q_lower for word in math_keywords)
            has_numbers = any(char.isdigit() for char in question)
            
            # Only reject if clearly not math-related
            if not has_math_keyword and not has_numbers:
                # Check if question is very generic
                generic_words = ["what is", "explain", "tell me", "describe"]
                if any(gen in q_lower for gen in generic_words) and len(question.split()) < 10:
                    response = f"Sorry, I can only answer Math questions."
                    self.memory.chat_memory.add_ai_message(response)
                    return response
        
        if self.subject.lower() == "english":
            english_keywords = ["read", "reading", "grammar", "vocabulary", "writing", "poem", "literature", "word", "sentence", "story", "noun", "verb", "adjective", "paragraph"]
            if not any(word in q_lower for word in english_keywords):
                response = f"Sorry, I can only answer English questions."
                self.memory.chat_memory.add_ai_message(response)
                return response
        
        if self.subject.lower() == "evs":
            evs_keywords = ["environment", "plant", "animal", "season", "pollution", "conservation", "resource", "nature", "earth", "water", "air", "living", "non-living"]
            if not any(word in q_lower for word in evs_keywords):
                response = f"Sorry, I can only answer EVS questions."
                self.memory.chat_memory.add_ai_message(response)
                return response
        
        # Try to solve math directly
        eq_solution = self._solve_equation(question)
        if eq_solution:
            self.memory.chat_memory.add_ai_message(eq_solution)
            return eq_solution
        
        # ✅ NEW: MMR retrieval with MORE candidates and DEBUG output
        print(f"\n[DEBUG] Question: {question}")
        
        retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,              # ✅ Increased from 3 to 5 for better coverage
                "fetch_k": 20,       # ✅ Increased from 10 to 20 for broader search
                "lambda_mult": 0.5   # ✅ More diversity (0.5) to get varied chunks
            }
        )
        docs = retriever.invoke(question)
        
        # ✅ DEBUG: Print retrieved chunks
        print(f"[DEBUG] Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs):
            print(f"[DEBUG] Chunk {i+1} (length: {len(doc.page_content)}): {doc.page_content[:200]}...")
        
        # Check if documents are meaningful (relaxed threshold)
        if not docs or all(len(d.page_content.strip()) < 30 for d in docs):  # ✅ Lowered from 50 to 30
            answer = f"Sorry, I don't have information on this topic in the {self.subject} textbook."
            self.memory.chat_memory.add_ai_message(answer)
            return answer
        
        # Build context + history
        context = "\n\n".join([d.page_content for d in docs])  # ✅ Double newline for better separation
        history = "\n".join([f"{m['type']}: {m['content']}" for m in self.memory.chat_memory.messages[-10:]])
        
        print(f"[DEBUG] Context length: {len(context)} characters")
        
        # ✅ IMPROVED: Less strict prompt for better responses
        prompt = f"""
You are a helpful {self.subject} teacher for elementary school students.

INSTRUCTIONS:
1. Read the Textbook Context below carefully
2. Answer the question using ONLY information from the Textbook Context
3. If you find the answer in the context, explain it clearly in simple language
4. If the exact answer is NOT in the context, say: "I cannot find this information in the textbook."
5. Use examples from the textbook when available
6. Keep your answer suitable for elementary students

Textbook Context:
{context}

Chat History:
{history}

Student Question:
{question}

Your Answer:
"""
        
        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, "content"):
                response = response.content
            response = str(response).strip()
            
            print(f"[DEBUG] LLM Response length: {len(response)} characters")
            print(f"[DEBUG] Response preview: {response[:200]}...")
            
            # ✅ RELAXED: Less aggressive hallucination detection
            if len(response) > 400 and "cannot find" not in response.lower():
                # Check if response has reasonable overlap with context
                context_lower = context.lower()
                response_lower = response.lower()
                
                # Extract meaningful words (more than 3 characters)
                response_words = set([w for w in response_lower.split() if len(w) > 3])
                context_words = set([w for w in context_lower.split() if len(w) > 3])
                
                if response_words and context_words:
                    overlap = len(response_words & context_words)
                    overlap_ratio = overlap / len(response_words) if response_words else 0
                    
                    print(f"[DEBUG] Word overlap ratio: {overlap_ratio:.2%}")
                    
                    # ✅ RELAXED: Lowered from 15% to 10%
                    if overlap_ratio < 0.10:
                        response = f"I cannot find this information in the {self.subject} textbook."
        
        except Exception as e:
            print(f"[DEBUG] Error: {e}")
            response = f"Error from model: {e}"
        
        # Save AI response into memory
        self.memory.chat_memory.add_ai_message(response)
        return response
