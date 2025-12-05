import re
import numpy as np
from langchain_ollama import OllamaEmbeddings
from core.memory import PersistentMemory
from agents.base_agent import BaseAgent

class MetaAgent:
    def __init__(self, agents):
        self.agents = agents
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Subject reference texts for routing
        subject_texts = {
            "Math": "algebra, geometry, arithmetic, fractions, equations, problem solving, numbers, calculation, formulas",
            "English": "grammar, vocabulary, reading, writing, comprehension, literature, stories, poems, sentences",
            "EVS": "environment, plants, animals, seasons, weather, community, safety, nature, conservation, resources"
        }
        
        self.subject_vectors = {
            k: self.embeddings.embed_query(v) for k, v in subject_texts.items()
        }
    
    def _split_questions(self, text: str):
        """
        Split multi-part questions into smaller sub-questions.
        Splits on ' and ', '?', or newlines.
        """
        parts = re.split(r"\?|\band\b|\n", text)
        return [p.strip() for p in parts if p.strip()]
    
    def _find_subject(self, question: str):
        """Find the best subject for a sub-question with confidence check."""
        q_vec = self.embeddings.embed_query(question)
        sims = {
            subj: np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec))
            for subj, vec in self.subject_vectors.items()
        }
        
        # Pick subject with highest similarity
        best_subject = max(sims, key=sims.get)
        confidence = sims[best_subject]
        
        # ✅ Explicit keyword overrides (highest priority)
        q_lower = question.lower()
        if "math" in q_lower or "solve" in q_lower or "equation" in q_lower or "calculate" in q_lower:
            return "Math", 1.0
        elif "english" in q_lower or "read" in q_lower or "grammar" in q_lower or "literature" in q_lower:
            return "English", 1.0
        elif "evs" in q_lower or "plant" in q_lower or "environment" in q_lower or "season" in q_lower:
            return "EVS", 1.0
        
        # ✅ NEW: Low confidence handling
        if confidence < 0.5:  # Ambiguous question
            print(f"[MetaAgent] Low confidence ({confidence:.2f}) for: '{question}'")
            print(f"[MetaAgent] Similarity scores: {sims}")
            # Try best guess anyway
            return best_subject, confidence
        
        return best_subject, confidence
    
    def route(self, question: str):
        sub_questions = self._split_questions(question)
        answers = []
        
        for sub_q in sub_questions:
            subject, confidence = self._find_subject(sub_q)
            
            # Check if subject agent exists
            if subject not in self.agents:
                answers.append(f"**Error**: {subject} subject not available. Please upload the PDF.")
                continue
            
            # ✅ Stateless: fresh memory for each query
            temp_agent = BaseAgent(
                subject,
                self.agents[subject].pdf_path,
                self.agents[subject].llm,
                PersistentMemory(subject)
            )
            
            ans = temp_agent.query(sub_q)
            
            # ✅ Show confidence in meta agent response
            confidence_emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
            answers.append(f"{confidence_emoji} **{subject} Answer** (confidence: {confidence:.0%}):\n{ans}")
        
        return "\n\n---\n\n".join(answers)
