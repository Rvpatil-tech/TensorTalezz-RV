import pytest
from tensor_talezz_rv.education import explain, quiz

def test_explain():
    info = explain("overfitting")
    assert info is not None
    assert "title" in info
    assert info["title"] == "Overfitting"
    
def test_quiz(monkeypatch):
    # Mock input to always answer correctly for the first 2 questions
    def mock_input(prompt):
        return "B" # many questions have B as answer
    monkeypatch.setattr('builtins.input', mock_input)
    
    score = quiz(n_questions=2, shuffle=False)
    # The first 2 questions in the bank both have answer "B"
    assert score == 2
