# Trợ lý tra cứu Nghị định (KB + RAG nhẹ)

Chatbot tra cứu các PDF trong `data/`, phù hợp hỏi mức phạt hoặc nội dung điều khoản. Hệ thống dùng TF-IDF để tìm đoạn liên quan (không cần GPU/model embedding lớn), sau đó tùy chọn dùng Groq để viết lại câu trả lời có kiểm soát theo các đoạn đã tìm thấy. Nếu chưa có API key, nó hiển thị trích đoạn nguồn thay vì tự tạo câu trả lời.

Các PDF hiện có là bản scan, nên lúc lập chỉ mục ứng dụng dùng Tesseract OCR (Việt + Anh). Trên Windows cần cài [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki), chọn gói tiếng Việt nếu trình cài đặt có tùy chọn, rồi thêm `C:\Program Files\Tesseract-OCR` vào `PATH`; trên Render dùng Dockerfile đã kèm Tesseract.

## Chạy bằng Jupyter Notebook

Mở `rag_nghi_dinh.ipynb`, chạy lần lượt các ô. Notebook sẽ cài thư viện (nếu cần), tạo `storage/rag_index.joblib`, và có thể khởi chạy giao diện Streamlit.

## Chạy cục bộ

```powershell
python -m pip install -r requirements.txt
python indexer.py
streamlit run app.py
```

Vào `http://localhost:8501`. Hãy sao chép `.env.example` thành `.env` và thiết lập `GROQ_API_KEY` nếu muốn dùng LLM. Streamlit không tự đọc `.env`; trên local hãy đặt biến môi trường PowerShell: `$env:GROQ_API_KEY="..."`.

## Deploy Render

1. Đẩy toàn bộ mã nguồn **và ba PDF trong `data/`** lên GitHub (không đẩy `.env`).
2. Vào Render → **New +** → **Web Service**, kết nối repository.
3. Chọn Runtime **Docker** (để có Tesseract OCR) và dùng `Dockerfile` đã có.
4. Thêm Environment Variable `GROQ_API_KEY` (tùy chọn) và deploy. Render tự cấp URL, giao diện đã responsive cho desktop/mobile.

`storage/chat_history.db` lưu lịch sử cục bộ. Trên Render free, file này có thể mất sau redeploy/restart do filesystem tạm thời. Để lịch sử bền vững, thay SQLite bằng Postgres/Supabase hoặc một persistent disk; không lưu API key trong repository.

## Lưu ý chất lượng

- Khi PDF là bản scan không có text, cần OCR trước khi lập chỉ mục.
- Câu trả lời có nguồn chỉ hỗ trợ tra cứu; cần mở PDF gốc để áp dụng pháp lý chính thức.
