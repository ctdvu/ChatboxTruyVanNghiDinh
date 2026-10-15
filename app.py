
from pathlib import Path
from datetime import datetime
import os,re,sqlite3,joblib,pdfplumber,requests,streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
R=Path(__file__).parent; P=R/'data'; S=R/'storage'; I=S/'rag_index.joblib'; D=S/'history.db'
def pieces(t):
 t=re.sub(r'\s+',' ',t or '').strip();return [t[i:i+900] for i in range(0,len(t),760) if len(t[i:i+900])>80]
def index():
 a=[]
 for f in P.glob('*.pdf'):
  with pdfplumber.open(f) as pdf:
   for p,o in enumerate(pdf.pages,1):a += [{'text':x,'file':f.name,'page':p} for x in pieces(o.extract_text())]
 v=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True);z={'v':v,'m':v.fit_transform([x['text'] for x in a]),'a':a};S.mkdir(exist_ok=True);joblib.dump(z,I);return z
@st.cache_resource
def load():
 if I.exists():
  z = joblib.load(I)
 else:
  from indexer import build_index
  z = build_index(output=I)
 # Chỉ mục tạo từ notebook dùng tên vectorizer/matrix/records.
 if 'v' not in z:
  z = {'v': z['vectorizer'], 'm': z['matrix'],
       'a': [{**r, 'file': r.get('source', r.get('file', ''))} for r in z['records']]}
 return z
def get(q):
 z=load();s=cosine_similarity(z['v'].transform([q]),z['m']).ravel();return [{**z['a'][i],'score':s[i]} for i in s.argsort()[::-1][:4] if s[i]>0]
def ask(q,c):
 key=os.getenv('GROQ_API_KEY')
 if not key:return None
 ev='\n'.join(f'[{i+1}] {x["text"]}' for i,x in enumerate(c))
 msg=f'Chỉ trả lời bằng tiếng Việt từ tư liệu sau, không suy đoán. Nêu nguồn [1], [2].\nCâu hỏi: {q}\nTư liệu:\n{ev}'
 try:
  x=requests.post('https://api.groq.com/openai/v1/chat/completions',headers={'Authorization':f'Bearer {key}'},json={'model':os.getenv('GROQ_MODEL','llama-3.3-70b-versatile'),'temperature':.1,'messages':[{'role':'user','content':msg}]},timeout=40);x.raise_for_status();return x.json()['choices'][0]['message']['content']
 except requests.RequestException:return None
S.mkdir(exist_ok=True)
with sqlite3.connect(D) as x:x.execute('CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY,role TEXT,text TEXT,at TEXT)')
st.set_page_config('Trợ lý Nghị định','⚖️',layout='centered')
if 'chat' not in st.session_state:st.session_state.chat=[]
st.markdown('<style>.block-container{max-width:880px}@media(max-width:600px){.block-container{padding:1rem .7rem}}</style>',unsafe_allow_html=True)
st.title('⚖️ Trợ lý tra cứu Nghị định');st.caption('Tra cứu mức phạt, điều khoản và nội dung từ PDF đã nạp.')
with st.sidebar:
 if st.button('Lập lại chỉ mục'):index();load.clear();st.rerun()
 if st.button('Xóa hội thoại'):st.session_state.chat=[];st.rerun()
for m in st.session_state.chat:
 with st.chat_message(m['role']):st.markdown(m['text']);st.caption(m.get('src',''))
if q:=st.chat_input('Ví dụ: Không đội mũ bảo hiểm phạt bao nhiêu?'):
 st.session_state.chat.append({'role':'user','text':q})
 with st.chat_message('user'):st.write(q)
 with st.chat_message('assistant'):
  c=get(q);a=ask(q,c) or ('Không tìm thấy đoạn phù hợp.' if not c else 'Các trích đoạn liên quan nhất:\n\n'+'\n\n'.join(f'**[{i+1}]** {x["text"]}' for i,x in enumerate(c)))
  st.markdown(a);src=' • '.join(dict.fromkeys(f'{x["file"]} — trang {x["page"]}' for x in c));st.caption('Nguồn: '+src)
 st.session_state.chat.append({'role':'assistant','text':a,'src':src})
 with sqlite3.connect(D) as x:x.execute('INSERT INTO chats(role,text,at) VALUES(?,?,?)',('user',q,datetime.now().isoformat()));x.execute('INSERT INTO chats(role,text,at) VALUES(?,?,?)',('assistant',a,datetime.now().isoformat()))
