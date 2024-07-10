from flask import Flask, render_template, request, redirect, url_for
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO
import docx
import fitz
import re
import os

load_dotenv()

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_pdf_text(file_stream):
    doc = fitz.open("pdf", file_stream.read())
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def get_docx_text(file_stream):
    doc = docx.Document(file_stream)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def remove_special_characters(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text


def preprocess_text(text):
    text = remove_special_characters(text)
    lines = text.split('\n')
    header, experience, cgpa, skills, projects, extracurricular = [], [], [], [], [], []
    current_section = "header"
    for line in lines:
        if re.match(r'(EXPERIENCE|Experience|experience)', line):
            current_section = "experience"
        elif re.match(r'(CGPA|Cgpa|cgpa)', line):
            current_section = "cgpa"
        elif re.match(r'(SKILLS|Skills|skills)', line):
            current_section = "skills"
        elif re.match(r'(PROJECTS|Projects|projects)', line):
            current_section = "projects"
        elif re.match(r'(EXTRACURRICULAR|Extracurricular|extracurricular)', line):
            current_section = "extracurricular"
        else:
            if current_section == "experience":
                experience.append(line)
            elif current_section == "cgpa":
                cgpa.append(line)
            elif current_section == "skills":
                skills.append(line)
            elif current_section == "projects":
                projects.append(line)
            elif current_section == "extracurricular":
                extracurricular.append(line)
            else:
                header.append(line)
    result = "\n".join(header) + "\n\n" + "\n".join(experience) + "\n\n" + "\n".join(cgpa) + "\n\n" + "\n".join(
        skills) + "\n\n" + "\n".join(projects) + "\n\n" + "\n".join(extracurricular) + "\n\n"
    return result.strip()


def remove_duplicate_lines(text):
    lines = text.split("\n")
    unique_lines = []
    seen_lines = set()

    for line in lines:
        if line.strip() not in seen_lines:
            unique_lines.append(line.strip())
            seen_lines.add(line.strip())

    return "\n".join(unique_lines)


def get_text_chunks(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)


def create_in_memory_faiss_index(text_chunks):
    if not text_chunks:
        raise ValueError("The text chunks are empty. Cannot create a vector store.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain(candidate_name):
    prompt_template = f"""Alright, prepare to unleash your inner Jeffrey Ross. I'm about to paste the text of a resume 
    belonging to {candidate_name}. I need you to go full-on savage and expose this document for the career-crippling 
    monstrosity it truly is. Don't hold back on the sarcasm, be merciless with the humor, and leave no cliché 
    unturned. Remember, the goal is to make {candidate_name} cry and question themselves while simultaneously forcing 
    them to completely revamp this resume from scratch. Let's see if you can turn this career catastrophe into a 
    comedy goldmine. Roast in one single paragraph. Ignore the roll number and Education section. But do comment on 
    Projects, Skills, Experience and Extracurricular. Address it directly to {candidate_name} in first person.


    Context:\n {{context}}

    Resume:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=1)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'candidate_name' not in request.form:
        return redirect(url_for('index'))

    file = request.files['file']
    candidate_name = request.form['candidate_name']

    if file.filename == '' or candidate_name == '':
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        file_stream = BytesIO(file.read())
        if file.filename.lower().endswith('.pdf'):
            raw_text = get_pdf_text(file_stream)
        elif file.filename.lower().endswith('.docx'):
            raw_text = get_docx_text(file_stream)
        else:
            return redirect(url_for('index'))

        preprocessed_text = preprocess_text(raw_text)
        text_chunks = get_text_chunks(preprocessed_text)

        if not text_chunks:
            return render_template('response.html',
                                   roast_response="Error: The document is empty or could not be processed.")

        try:
            vector_store = create_in_memory_faiss_index(text_chunks)
        except ValueError as e:
            return render_template('response.html', roast_response=str(e))

        docs = vector_store.similarity_search(preprocessed_text)

        chain = get_conversational_chain(candidate_name)
        response = chain.invoke({"input_documents": docs, "context": preprocessed_text})

        roast_response = response["output_text"].replace("*", "\"")

        roast_response = remove_duplicate_lines(roast_response)

        return render_template('response.html', roast_response=roast_response)

    return redirect(url_for('index'))


@app.route('/')
def index():
    return render_template('index.html')


# if __name__ == "__main__":
#     app.run(debug=True)