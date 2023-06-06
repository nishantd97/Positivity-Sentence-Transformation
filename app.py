from functools import lru_cache

from flask import Flask, render_template, request
import os
from transformers import Trainer, pipeline, set_seed, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

app = Flask(__name__)
model_path = 'models'

model_unconstrained = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_path,'t5_unconstrained/'))
tokenizer_unconstrained = AutoTokenizer.from_pretrained(os.path.join(model_path,'t5_unconstrained/'))
reframer_unconstrained = pipeline('summarization', model=model_unconstrained, tokenizer=tokenizer_unconstrained)


model_controlled = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_path,'t5_controlled/'))
tokenizer_controlled = AutoTokenizer.from_pretrained(os.path.join(model_path,'t5_controlled/'))
reframer_controlled = pipeline('summarization', model=model_controlled, tokenizer=tokenizer_controlled)

model_predict = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(model_path,'t5_predict/'))
tokenizer_predict = AutoTokenizer.from_pretrained(os.path.join(model_path,'t5_predict/'))
reframer_predict = pipeline('summarization', model=model_predict, tokenizer=tokenizer_predict)

@app.route("/")
def landing_page():
    """
    Renders landing page
    """
    print("Hello world")
    return render_template("landing.html")


@app.route("/v1", methods=["POST", "GET"])
def v1():
    """
    Renders v1 model input form and results
    """
    return handle_text_request(request, "v1.html")


@app.route("/v2", methods=["POST", "GET"])
def v2():
    """
        Renders v2 model input form and results
        """
    return handle_text_request(request, "v2.html")


@app.route("/v3", methods=["POST", "GET"])
def v3():
    """
        Renders v3 model input form and results
    """
    return handle_text_request(request, "v3.html")


def get_model_from_template(template_name):
    """
    Get the name of the relevant model from the name of the template
    :param template_name: name of html template
    :return: name of the model
    """
    return template_name.split(".")[0]


@lru_cache(maxsize=128)
def retrieve_recommendations_for_model(question, model):
    """
    This function computes or retrieves recommendations
    We use an LRU cache to store results we process. If we see the same sentence
    twice, we can retrieve cached results to serve them faster
    :param question: the input text to the model
    :param model: which model to use
    :return: a model's recommendations
    """

    if model == "v1":
        reframed_phrase = reframer_unconstrained(question)[0]['summary_text']
        return reframed_phrase
    
    elif model == "v2":
        reframed_phrase = reframer_controlled(question)[0]['summary_text']
        return reframed_phrase
    
    elif model == "v3":
        reframed_phrase = reframer_predict(question)[0]['summary_text']
        return reframed_phrase
    raise ValueError("Incorrect Model passed")


def handle_text_request(request, template_name):
    """
    Renders an input form for GET requests and displays results for the given
    posted question for a POST request
    :param request: http request
    :param template_name: name of the requested template (e.g. v2.html)
    :return: Render an input form or results depending on request type
    """
    if request.method == "POST":
        question = request.form.get("question")
        model_name = get_model_from_template(template_name)
        suggestions = retrieve_recommendations_for_model(question, model_name)
        payload = {
            "input": question,
            "suggestions": suggestions,
            "model_name": model_name,
        }
        return render_template("results.html", ml_result=payload)
    else:
        return render_template(template_name)
    

app.run(debug = True, host='0.0.0.0', port=5000)