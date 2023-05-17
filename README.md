# Hướng dẫn nhập môn LangChain bằng tiếng Việt
> Để thuận tiện cho việc đọc, đã tạo ra gitbook: https://? liaokong.gitbook.io/llm-kai-fa-jiao-cheng/

> Đã thêm CHANGELOG, nếu có nội dung mới, tôi sẽ viết ở đây để bạn có thể dễ dàng xem nội dung cập nhật mới

> Nếu bạn muốn thay đổi địa chỉ gốc yêu cầu API OPENAI thành địa chỉ proxy của riêng mình, bạn có thể thay đổi bằng cách đặt biến môi trường "OPENAI_API_BASE".
> 
> Mã tham khảo liên quan: https://github.com/openai/openai-python/blob/d6fa3bfaae69d639b0dd2e9251b375d7070bbef1/openai/__init__.py#L48

## Giới thiệu
Như chúng ta đã biết, API của OpenAI không thể kết nối mạng, vì vậy nếu chỉ sử dụng chức năng của chính mình để tìm kiếm trực tuyến và cung cấp câu trả lời, tóm tắt tài liệu PDF, trả lời câu hỏi dựa trên video YouTube, vv, thì chắc chắn là không thể thực hiện được. Vì vậy, chúng ta sẽ giới thiệu một thư viện nguồn mở rất mạnh mẽ: LangChain.

> Tài liệu: https://python.langchain.com/en/latest/

Thư viện này hiện đang rất sôi động, được cập nhật hàng ngày và đã có 22k ngôi sao, tốc độ cập nhật nhanh chóng.

LangChain là một framework được sử dụng để phát triển ứng dụng được định hướng bởi mô hình ngôn ngữ. Nó có hai khả năng chính:

1. Kết nối mô hình LLM với nguồn dữ liệu bên ngoài.
2. Tương tác với mô hình LLM.
> Mô hình LLM: Large Language Model, Mô hình ngôn ngữ lớn

##

## Các chức năng cơ bản
> Gọi LLM

* Hỗ trợ nhiều giao diện mô hình, chẳng hạn như OpenAI, Hugging Face, AzureOpenAI ...
* Fake LLM, được sử dụng cho kiểm thử
* Hỗ trợ bộ nhớ cache, chẳng hạn in-mem (trong bộ nhớ), SQLite, Redis, SQL
* Ghi lại lượng sử dụng
* Hỗ trợ chế độ luồng (trả về từng ký tự một, tương tự hiệu ứng đánh máy)

Quản lý Prompt, hỗ trợ các mẫu tùy chỉnh

Có nhiều công cụ tải tài liệu, chẳng hạn như Email, Markdown, PDF, YouTube ...

Hỗ trợ chỉ mục

* Trình phân đoạn tài liệu
* Vector hóa
* Liên kết với lưu trữ và tìm kiếm vector, chẳng hạn Chroma, Pinecone, Qdrand

Chains

* LLMChain
* Các công cụ Chain khác nhau
* LangChainHub

## Khái niệm cần biết

Tôi tin rằng sau khi đọc phần giới thiệu ở trên, hầu hết mọi người sẽ cảm thấy mơ hồ. Đừng lo, những khái niệm trên thực tế không quá quan trọng khi bạn mới bắt đầu học, khi chúng ta đã hoàn thành các ví dụ phía sau, quay lại xem nội dung trên sẽ hiểu rõ hơn.

Tuy nhiên, có một số khái niệm bạn phải biết.

##

### Loader (Trình tải)

Như tên gọi, đây là công cụ dùng để tải dữ liệu từ nguồn cụ thể. Ví dụ: `DirectoryLoader` (trình tải thư mục), `AzureBlobStorageContainerLoader` (trình tải lưu trữ Azure Blob), `CSVLoader` (trình tải file CSV), `EverNoteLoader` (trình tải từ EverNote), `GoogleDriveLoader` (trình tải từ Google Drive), `UnstructuredHTMLLoader` (trình tải từ trang web bất kỳ), `PyPDFLoader` (trình tải từ file PDF), `S3DirectoryLoader/S3FileLoader` (trình tải từ dịch vụ lưu trữ S3), `YoutubeLoader` (trình tải từ YouTube), và còn nhiều trình tải khác nữa. Các trình tải này đều được cung cấp bởi thư viện LangChain.

> Tài liệu: https://python.langchain.com/en/latest/modules/indexes/document_loaders.html

### 

### Document (Tài liệu)

Sau khi dữ liệu được tải lên bằng trình tải, nó cần được chuyển đổi thành đối tượng Document trước khi có thể sử dụng cho các mục đích tiếp theo.

###

### Text Splitters (Trình tách văn bản)

Như tên gọi, trình tách văn bản được sử dụng để tách văn bản thành các phần nhỏ. Tại sao chúng ta cần phân tách văn bản? Bởi vì khi gửi văn bản làm prompt cho API của OpenAI hoặc sử dụng tính năng nhúng của OpenAI API, chúng ta luôn có giới hạn về số ký tự.

Ví dụ, nếu chúng ta gửi một tệp PDF gồm 300 trang cho API của OpenAI để tóm tắt, chắc chắn nó sẽ báo lỗi vượt quá giới hạn số ký tự tối đa. Do đó, chúng ta cần sử dụng trình tách văn bản để phân tách Document đã tải từ trình tải.

###

### Vectorstores (Cơ sở dữ liệu vector)

Vì tìm kiếm liên quan đến dữ liệu thực chất là hoạt động vector, nên khi sử dụng tính năng nhúng của OpenAI API hoặc truy vấn trực tiếp từ cơ sở dữ liệu vector, chúng ta cần chuyển đổi `Document` đã tải thành vector để thực hiện tìm kiếm vector. Việc chuyển đổi thành vector cũng rất đơn giản, chỉ cần lưu trữ dữ liệu vào cơ sở dữ liệu vector tương ứng.

Thư viện LangChain cũng cung cấp nhiều cơ sở dữ liệu vector cho chúng ta sử dụng.

> Tài liệu: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html

###

### Chain (Dây chuyền)

Chúng ta có thể hiểu Chain như một nhiệm vụ. Một Chain là một nhiệm vụ, và tất nhiên, chúng ta cũng có thể thực hiện nhiều nhiệm vụ như một dây chuyền.

###

### Agent (Đại lý)

Chúng ta có thể hiểu Agent như một người đại diện có khả năng chọn lựa và gọi Chain hoặc công cụ đã có một cách linh hoạt.

Quá trình thực thi có thể tham khảo hình ảnh dưới đây:

![image-20230406213322739](doc/image-20230406213322739.png)

### Embedding (Nhúng)

Dùng để đo lường mức độ liên quan giữa các đoạn văn bản. Đây cũng là yếu tố quan trọng nhất khi xây dựng cơ sở kiến thức của riêng mình bằng OpenAI API.

So với fine-tuning, ưu điểm lớn nhất của nhúng là không cần huấn luyện và có thể thêm nội dung mới theo thời gian thực mà không cần huấn luyện lại từ đầu, đồng thời chi phí thấp hơn về mọi mặt so với fine-tuning.

> Bạn có thể tham khảo video này để so sánh và lựa chọn cụ thể: https://www.youtube.com/watch?v=9qq6HTr7Ocw

## 

## Thực hành
Sau khi đã hiểu các khái niệm cần thiết ở trên, có thể bạn vẫn còn mơ hồ. Điều này không quan trọng, tôi tin sau khi hoàn thành các ví dụ phía sau, bạn sẽ hiểu rõ hơn về nội dung trên và cảm nhận được sức mạnh thực sự của thư viện này.

Vì chúng ta đang đi sâu vào việc nâng cao OpenAI API, nên các ví dụ sau đây sẽ sử dụng LLM của OpenAI làm ví dụ. Sau đó, bạn có thể thay thế bằng mô hình LLM mà bạn cần dựa trên nhiệm vụ của mình.

Tất nhiên, ở cuối bài viết này, toàn bộ mã sẽ được lưu trữ dưới dạng tệp colab ipynb để bạn có thể học tập.

> Đề nghị bạn tuần tự xem từng ví dụ, vì ví dụ tiếp theo sẽ sử dụng những điểm kiến thức đã được giới thiệu trong ví dụ trước.
>
> Tuy nhiên, nếu bạn không hiểu một số điều, không cần lo lắng, bạn có thể tiếp tục đọc, vì trong lần học đầu tiên, không cần phải hiểu rõ tất cả.


```python
import os
os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'
os.environ["SERPAPI_API_KEY"] = 'YOUR_API_KEY'
```

Sau đó, bạn có thể bắt đầu viết code của mình.

```python
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.agents import AgentType

# Tải mô hình OpenAI
llm = OpenAI(temperature=0, max_tokens=2048)

# Tải công cụ serpapi
tools = load_tools(["serpapi"])

# Nếu sau khi tìm kiếm bạn muốn tính toán một số thứ khác, bạn có thể viết như sau
# tools = load_tools(['serpapi', 'llm-math'], llm=llm)

# Nếu sau khi tìm kiếm bạn muốn sử dụng lệnh `print` trong Python để thực hiện tính toán đơn giản, bạn có thể viết như sau
# tools = load_tools(["serpapi","python_repl"])

# Sau khi tải công cụ, cần khởi tạo chúng. Tham số verbose=True sẽ hiển thị thông tin chi tiết về quá trình thực thi.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Chạy agent
agent.run("What's the date today? What great events have taken place today in history?")
```
![image-20230404234236982](doc/image-20230404234236982.png)

Chúng ta có thể thấy rằng agent đã trả về ngày hôm nay (dựa trên múi giờ) và các sự kiện quan trọng trong lịch sử ngày hôm nay.

Cả chain và agent đều có tham số `verbose`, đây là một tham số rất hữu ích, khi bật nó sẽ hiển thị quá trình thực thi của chain đầy đủ.

Trong kết quả trả về ở trên, chúng ta có thể thấy rằng nó đã phân tách câu hỏi của chúng ta thành các bước và từng bước dẫn đến kết quả cuối cùng.

Về các loại agent (agent type) có thể hiểu như sau (nếu bạn không hiểu, không sao cả, việc này không ảnh hưởng đến việc học tiếp phía dưới):

* zero-shot-react-description: Dựa vào mô tả của công cụ và nội dung yêu cầu để quyết định sử dụng công cụ nào (phổ biến nhất).
* react-docstore: Tương tác với ReAct framework và docstore, sử dụng công cụ Search và Lookup, ví dụ: công cụ Wikipedia.
* self-ask-with-search: Agent này chỉ sử dụng một công cụ: Intermediate Answer, nó sẽ tìm các câu trả lời dựa trên thực tế (điều đó có nghĩa là không phải câu trả lời được tạo bởi GPT, mà là câu trả lời đã tồn tại trong mạng lưới, văn bản, ví dụ: công cụ Google search API.
* conversational-react-description: Được thiết kế cho các tương tác hội thoại, prompt của nó được thiết kế có tính liên quan đến hội thoại và vẫn sử dụng ReAct framework để quyết định sử dụng công cụ nào, và lưu trữ tương tác trước trong bộ nhớ.
Bạn có thể đọc về ReAct tại đây: https://arxiv.org/pdf/2210.03629.pdf

> Thực hiện ReAct mode của LLM bằng Python: https://til.simonwillison.net/llms/python-react-pattern
>
> Giải thích về agent type:
>
> https://python.langchain.com/en/latest/modules/agents/agents/agent\_types.html?highlight=zero-shot-react-description

> Một điểm cần lưu ý là `serpapi` có vẻ không hỗ trợ tiếng Việt tốt, vì vậy, đề nghị sử dụng tiếng Anh trong prompt.

Tất nhiên, OpenAI đã viết sẵn agent cho `ChatGPT Plugins`, trong tương lai, khi ChatGPT có thể sử dụng các plugin, chúng ta cũng có thể sử dụng các plugin trong API, nghe đã thấy vui rồi.

Tuy nhiên, hiện tại chỉ có thể sử dụng các plugin không yêu cầu xác thực, hy vọng trong tương lai OpenAI sẽ giải quyết vấn đề này.

Bạn có thể xem tài liệu này để biết thêm chi tiết: https://python.langchain.com/en/latest/modules/agents/tools/examples/chatgpt\_plugins.html

> ChatGPT chỉ giúp OpenAI kiếm tiền, trong khi OpenAI API giúp tôi kiếm tiền.

### Đối với việc tóm tắt văn bản dài

Khi chúng ta muốn sử dụng API của OpenAI để tóm tắt một đoạn văn bản, cách thông thường là gửi nó trực tiếp cho API để tóm tắt. Tuy nhiên, nếu đoạn văn bản vượt quá giới hạn token tối đa của API, sẽ xảy ra lỗi.

Trong trường hợp này, chúng ta thường chia đoạn văn thành các phần nhỏ, ví dụ bằng cách sử dụng tiktoken để tính toán và chia nhỏ, sau đó gửi từng phần cho API để tóm tắt, cuối cùng tóm tắt từng phần lại thành một tóm tắt toàn bộ.

Nếu bạn sử dụng LangChain, nó đã giúp chúng ta xử lý quy trình này một cách tốt, làm cho việc viết mã trở nên rất đơn giản.

Không nói nhiều, chúng ta đi thẳng vào mã.

```python
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

# Tải văn bản
loader = UnstructuredFileLoader("/content/sample_data/data/lg_test.txt")
# Chuyển văn bản thành đối tượng Document
document = loader.load()
print(f"Số lượng documents: {len(document)}")

# Khởi tạo công cụ chia văn bản
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0
)

# Chia nhỏ văn bản
split_documents = text_splitter.split_documents(document)
print(f"Số lượng documents sau khi chia nhỏ: {len(split_documents)}")

# Tải mô hình llm
llm = OpenAI(model_name="text-davinci-003", max_tokens=1500)

# Tạo chuỗi tóm tắt
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# Chạy chuỗi tóm tắt (chỉ tóm tắt 5 đoạn để minh họa nhanh)
chain.run(split_documents[:5])
```

Trước tiên, chúng ta in ra số lượng documents trước và sau khi chia nhỏ, ta có thể thấy ban đầu chỉ có một document duy nhất, sau khi chia nhỏ, document được chia thành 317 document.

![image-20230405162631460](doc/image-20230405162631460.png)

Kết quả cuối cùng là tóm tắt cho 5 document đầu tiên.

![image-20230405162937249](doc/image-20230405162937249.png)

Dưới đây là một số tham số cần lưu ý:

**Tham số chunk_overlap của công cụ chia văn bản**

Tham số này xác định số lượng ký tự cuối cùng của document trước được bao gồm trong từng document sau khi chia nhỏ, nó có tác dụng tăng liên kết ngữ cảnh giữa các document. Ví dụ, khi `chunk_overlap=0`, document đầu tiên là aaaaaa, document thứ hai là bbbbbb; khi `chunk_overlap=2`, document đầu tiên là aaaaaa, document thứ hai là aabbbbbb.

Tuy nhiên, điều này không tuyệt đối và phụ thuộc vào thuật toán cụ thể trong mô hình chia văn bản mà bạn sử dụng.

> Bạn có thể tham khảo tài liệu về công cụ chia văn bản tại đây: https://python.langchain.com/en/latest/modules/indexes/text\_splitters.html

**Tham số `chain_type` của chuỗi tóm tắt**

Tham số này điều khiển cách truyền document cho mô hình llm, có tổng cộng 4 cách:

`stuff`: Đây là cách đơn giản nhất, sẽ gửi tất cả các document cùng một lúc cho mô hình llm để tóm tắt. Nếu có quá nhiều document, sẽ xảy ra lỗi vượt quá giới hạn token tối đa, nên không thường được sử dụng trong tóm tắt văn bản.

`map_reduce`: Phương pháp này sẽ tóm tắt từng document trước, sau đó tóm tắt tất cả các kết quả tóm tắt của các document lại một lần nữa.

![image-20230405165752743](doc/image-20230405165752743.png)

`refine`: Phương pháp này sẽ tóm tắt document đầu tiên, sau đó gửi nội dung tóm tắt của document đầu tiên và document thứ hai cùng một lúc cho mô hình llm để tóm tắt, và tiếp tục như vậy. Phương pháp này có lợi thế là khi tóm tắt document tiếp theo, nó sẽ được tóm tắt kèm theo document trước đó, tăng tính liên tục của nội dung tóm tắt.

![image-20230405170617383](doc/image-20230405170617383.png)

`map_rerank`: Phương pháp này thường không được sử dụng trong chuỗi tóm tắt, mà thường được sử dụng trong chuỗi trả lời câu hỏi. Đầu tiên, bạn cần cung cấp một câu hỏi, nó sẽ tính điểm xác suất mỗi document có thể trả lời câu hỏi đó, sau đó tìm document có điểm cao nhất và gửi document đó cùng với câu hỏi để mô hình llm trả về câu trả lời cụ thể.

### Xây dựng chatbot hỏi đáp dựa trên cơ sở tri thức cục bộ

Trong ví dụ này, chúng ta sẽ hướng dẫn cách xây dựng một cơ sở tri thức bằng cách đọc nhiều tài liệu từ nguồn cục bộ và sử dụng OpenAI API để tìm kiếm và cung cấp câu trả lời trong cơ sở tri thức.

Đây là một hướng dẫn rất hữu ích, ví dụ: bạn có thể dễ dàng tạo ra một chatbot để giới thiệu về hoạt động kinh doanh của công ty hoặc giới thiệu về một sản phẩm cụ thể.

```python
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

# Tải tất cả các tệp có định dạng txt từ thư mục
loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
# Chuyển đổi dữ liệu thành đối tượng document, mỗi tệp sẽ trở thành một document
documents = loader.load()

# Khởi tạo công cụ chia văn bản
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# Chia nhỏ document đã tải
split_docs = text_splitter.split_documents(documents)

# Khởi tạo đối tượng embeddings của OpenAI
embeddings = OpenAIEmbeddings()
# Tính toán vector nhúng và lưu trữ tạm thời thông tin vector của document vào cơ sở dữ liệu vector Chroma để sử dụng trong truy vấn khớp
docsearch = Chroma.from_documents(split_docs, embeddings)

# Tạo đối tượng hỏi đáp
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
# Tiến hành hỏi đáp
result = qa({"query": "Tổng doanh thu của iFLYTEK trong quý đầu tiên của năm nay là bao nhiêu?"})
print(result)
```

![image-20230405173730382](doc/image-20230405173730382.png)

Chúng ta có thể thấy từ kết quả rằng nó thành công lấy ra câu trả lời chính xác từ dữ liệu chúng ta đã cung cấp.

> Thông tin chi tiết về OpenAI embeddings có thể được xem tại liên kết này: https://platform.openai.com/docs/guides/embeddings

### Xây dựng cơ sở dữ liệu chỉ mục vector

Trong ví dụ trước đó, chúng ta đã có một bước là chuyển đổi thông tin document thành thông tin vector và nhúng, sau đó lưu trữ tạm thời vào cơ sở dữ liệu Chroma.

Tuy nhiên, vì đó chỉ là lưu trữ tạm thời, khi chúng ta hoàn thành việc thực thi mã trên, dữ liệu đã được nhúng sẽ bị mất đi. Nếu muốn sử dụng lại lần sau, chúng ta sẽ cần tính toán lại nhúng, điều này chắc chắn không phải là điều chúng ta muốn.

Vì vậy, trong ví dụ này, chúng ta sẽ giải thích cách lưu trữ dữ liệu vector một cách lâu dài thông qua hai cơ sở dữ liệu Chroma và Pinecone.

> Vì LangChain hỗ trợ nhiều cơ sở dữ liệu, nên ở đây chúng tôi chỉ giới thiệu hai cơ sở dữ liệu được sử dụng phổ biến hơn, bạn có thể xem thêm tài liệu tại đây: https://python.langchain.com/en/latest/modules/indexes/vectorstores/getting\_started.html

**Chroma**

Chroma là một cơ sở dữ liệu vector cục bộ, nó cung cấp phương thức `persist_directory` để thiết lập thư mục lưu trữ dữ liệu lâu dài. Khi đọc, chỉ cần sử dụng phương thức from_document để tải cơ sở dữ liệu.

```python
from langchain.vectorstores import Chroma

# Lưu trữ dữ liệu
docsearch = Chroma.from_documents(documents, embeddings, persist_directory="D:/vector_store")
docsearch.persist()

# Tải cơ sở dữ liệu
docsearch = Chroma(persist_directory="D:/vector_store", embedding_function=embeddings)
```

**Pinecone**

Pinecone là một cơ sở dữ liệu vector trực tuyến. Do đó, bước đầu tiên, bạn cần đăng ký tài khoản và nhận khóa API tương ứng từ https://app.pinecone.io/.

> Phiên bản miễn phí sẽ bị xóa tự động sau 14 ngày nếu không sử dụng chỉ mục.

Tiếp theo, hãy tạo cơ sở dữ liệu của chúng ta:

Tên chỉ mục: Tuỳ ý

Số chiều: Mô hình text-embedding-ada-002 của OpenAI có kích thước OUTPUT DIMENSIONS là 1536, vì vậy ta điền 1536 ở đây.

Phương pháp đo: Có thể mặc định là cosine

Loại Pod: S1 nếu bạn muốn tiết kiệm, P1 nếu bạn muốn tăng tốc độ

![image-20230405184646314](doc/image-20230405184646314.png)

Lưu trữ dữ liệu và mã tải dữ liệu như sau

```python
# Lưu trữ dữ liệu
docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)

# Tải dữ liệu
docsearch = Pinecone.from_existing_index(index_name, embeddings)
```

Dưới đây là phiên bản dịch tiếng Việt theo định dạng file MD5:

​```python
# Lưu trữ dữ liệu
docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)

# Tải dữ liệu
docsearch = Pinecone.from_existing_index(index_name, embeddings)
​```

Và đây là một ví dụ đơn giản về mã lấy embeddings từ cơ sở dữ liệu và trả lời:

```python
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

import pinecone

# Khởi tạo Pinecone
pinecone.init(
  api_key="YOUR_API_KEY",
  environment="YOUR_ENVIRONMENT"
)

loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
# Chuyển đổi dữ liệu thành đối tượng document, mỗi tệp sẽ trở thành một document
documents = loader.load()

# Khởi tạo công cụ chia văn bản
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
# Chia nhỏ document đã tải
split_docs = text_splitter.split_documents(documents)

index_name = "liaokong-test"

# Lưu trữ dữ liệu
# docsearch = Pinecone.from_texts([t.page_content for t in split_docs], embeddings, index_name=index_name)

# Tải dữ liệu
docsearch = Pinecone.from_existing_index(index_name, embeddings)

query = "Tổng doanh thu của iFLYTEK trong quý đầu tiên của năm nay là bao nhiêu?"
docs = docsearch.similarity_search(query, include_metadata=True)

llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
chain.run(input_documents=docs, question=query)
```

![image-20230407001803057](doc/image-20230407001803057.png)

### Xây dựng chatbot trả lời câu hỏi trên kênh YouTube sử dụng mô hình GPT3.5

Sau khi GPT-3.5-Turbo được công bố, nó nhận được sự yêu thích vì tính linh hoạt và hiệu suất tốt. Vì vậy, LangChain cũng đã cung cấp một loạt các chuỗi và mô hình riêng cho GPT-3.5-Turbo. Hãy xem ví dụ sau để biết cách sử dụng nó.

```python
import os

from langchain.document_loaders import YoutubeLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
ChatPromptTemplate,
SystemMessagePromptTemplate,
HumanMessagePromptTemplate
)

# Tải dữ liệu từ kênh YouTube
loader = YoutubeLoader.from_youtube_channel('https://www.youtube.com/watch?v=Dj60HHy-Kqk')

# Chuyển đổi dữ liệu thành đối tượng document
documents = loader.load()

# Khởi tạo công cụ chia văn bản
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=1000,
chunk_overlap=20
)

# Chia nhỏ các tài liệu YouTube
documents = text_splitter.split_documents(documents)

# Khởi tạo embeddings OpenAI
embeddings = OpenAIEmbeddings()

# Lưu trữ dữ liệu trong cơ sở dữ liệu vector
vector_store = Chroma.from_documents(documents, embeddings)

# Khởi tạo trình tìm kiếm từ cơ sở dữ liệu vector
retriever = vector_store.as_retriever()

system_template = """
Sử dụng ngữ cảnh sau để trả lời câu hỏi của người dùng.
Nếu bạn không biết câu trả lời, hãy nói rằng bạn không biết, đừng cố tạo câu trả lời. Và trả lời bằng tiếng Việt.
{context}
{chat_history}
"""

# Xây dựng danh sách tin nhắn ban đầu, tương tự như tham số `messages` được truyền cho OpenAI
messages = [  SystemMessagePromptTemplate.from_template(system_template),  HumanMessagePromptTemplate.from_template('{question}')]

# Khởi tạo đối tượng prompt
prompt = ChatPromptTemplate.from_messages(messages)


# Khởi tạo chuỗi Question-Answering
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1,max_tokens=2048),retriever,qa_prompt=prompt)


chat_history = []
while True:
  question = input('Câu hỏi:')
  # Bắt đầu gửi câu hỏi, chat_history là tham số bắt buộc, được sử dụng để lưu lịch sử trò chuyện
  result = qa({'question': question, 'chat_history': chat_history})
  chat_history.append((question, result['answer']))
  print(result['answer'])
```

Chúng ta có thể thấy rằng nó có thể trả lời câu hỏi xung quanh video YouTube một cách chính xác.

![image-20230406211923672](doc/image-20230406211923672.png)

Nó cũng thuận tiện để sử dụng câu trả lời trực tuyến

```python
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

chat = ChatOpenAI(streaming=True, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]), verbose=True, temperature=0)
resp = chat(chat_prompt_with_values.to_messages())
```

### Kết nối các công cụ với OpenAI bằng Zapier

Chúng ta chủ yếu kết hợp sử dụng `Zapier` để kết nối các công cụ lại với nhau.

Vì vậy, bước đầu tiên vẫn là đăng ký tài khoản và nhận khóa API tự nhiên của Zapier. https://zapier.com/l/natural-language-actions

Mặc dù khóa API của nó yêu cầu bạn điền thông tin, nhưng sau khi điền thông tin, bạn có thể nhìn thấy email thông báo được chấp thuận trong hộp thư của mình.

Sau đó, chúng ta mở cấu hình API của mình bằng cách nhấp chuột phải vào liên kết trong đó. Chúng ta nhấp vào "Quản lý các Hành động" bên phải để cấu hình các ứng dụng chúng ta muốn sử dụng.

Tôi đã cấu hình hành động đọc và gửi email của Gmail và tất cả các trường đều được chọn là "Dự đoán qua AI".

![image-20230406233319250](doc/image-20230406233319250.png)

![image-20230406234827815](doc/image-20230406234827815.png)


Sau khi cấu hình xong, chúng ta bắt đầu viết mã

```python
import os
os.environ["ZAPIER_NLA_API_KEY"] = ''
```
```python
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper


llm = OpenAI(temperature=.3)
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(toolkit.get_tools(), llm, agent="zero-shot-react-description", verbose=True)

# Chúng ta có thể in ra để xem chúng ta đã cấu hình các công cụ nào trong Zapier có thể sử dụng
for tool in toolkit.get_tools():
  print (tool.name)
  print (tool.description)
  print ("\n\n")

agent.run('Vui lòng tóm tắt email cuối cùng từ "******@qq.com" mà tôi đã nhận và gửi tóm tắt đó đến "******@qq.com"')
```

![image-20230406234712909](doc/image-20230406234712909.png)

Chúng ta có thể thấy rằng nó thành công đọc email cuối cùng từ `******@qq.com` và gửi nội dung tóm tắt đó đến `******@qq.com`

Đây là tin nhắn tôi đã gửi tới Gmail.

![image-20230406234017369](doc/image-20230406234017369.png)

Đây là email tôi gửi vào hộp thư QQ.

![image-20230406234800632](doc/image-20230406234800632.png)


Đây chỉ là một ví dụ nhỏ, vì `Zapier` có hàng ngàn ứng dụng, chúng ta có thể dễ dàng kết hợp OpenAI API để xây dựng quy trình làm việc của riêng mình.

## Một số ví dụ nhỏ

Một số chủ đề lớn đã được trình bày, phần còn lại là một số ví dụ nhỏ thú vị để mở rộng thêm.

### **Thực hiện nhiều chuỗi (chain) cùng một lúc**

Vì chúng được kết nối chuỗi, nên chúng ta có thể thực hiện nhiều chuỗi theo thứ tự tuần tự.

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain

# Chuỗi vị trí (location)
llm = OpenAI(temperature=1)
template = """Nhiệm vụ của bạn là tạo ra một món ăn cổ điển từ vùng mà người dùng gợi ý.
% VỊ TRÍ NGƯỜI DÙNG
{user_location}

CÂU TRẢ LỜI CỦA BẠN:
"""
prompt_template = PromptTemplate(input_variables=["user_location"], template=template)
location_chain = LLMChain(llm=llm, prompt=prompt_template)

# Chuỗi bữa ăn (meal)

template = """Với một bữa ăn, hãy đưa ra một công thức ngắn gọn và đơn giản để làm món ăn đó tại nhà.
% BỮA ĂN
{user_meal}

CÂU TRẢ LỜI CỦA BẠN:
"""
prompt_template = PromptTemplate(input_variables=["user_meal"], template=template)
meal_chain = LLMChain(llm=llm, prompt=prompt_template)

# Kết nối các chuỗi lại với nhau bằng SimpleSequentialChain, câu trả lời đầu tiên sẽ được thay thế bằng user_meal trong câu trả lời thứ hai, sau đó hỏi lại.
overall_chain = SimpleSequentialChain(chains=[location_chain, meal_chain], verbose=True)
review = overall_chain.run("Rome")
```
![image-20230406000133339](doc/image-20230406000133339.png)

### **Kết xuất dữ liệu có cấu trúc**

Đôi khi chúng ta muốn xuất ra nội dung không phải là văn bản mà là dữ liệu có cấu trúc giống như JSON.

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003")

# Chúng ta khai báo các trường dữ liệu cần xuất ra và kiểu dữ liệu của mỗi trường
response_schemas = [
    ResponseSchema(name="bad_string", description="Đây là một chuỗi đầu vào của người dùng không đúng định dạng"),
    ResponseSchema(name="good_string", description="Đây là câu trả lời của bạn, một câu trả lời được định dạng lại")
]

# Khởi tạo trình phân tích kết xuất dữ liệu có cấu trúc từ các schema trên
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Chuẩn bị thông báo định dạng
# {
#   "bad_string": string  // Đây là một chuỗi đầu vào của người dùng không đúng định dạng
#   "good_string": string  // Đây là câu trả lời của bạn, một câu trả lời được định dạng lại
#}
format_instructions = output_parser.get_format_instructions()

template = """
Bạn sẽ nhận được một chuỗi đầu vào không đúng định dạng từ người dùng.
Hãy định dạng lại nó và đảm bảo tất cả các từ đều được viết đúng

{format_instructions}

% ĐẦU VÀO NGƯỜI DÙNG:
{user_input}

CÂU TRẢ LỜI CỦA BẠN:
"""

# Nhúng mô tả định dạng vào trong prompt, thông báo cho llm biết chúng ta cần định dạng nội dung như thế nào
prompt = PromptTemplate(
    input_variables=["user_input"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)

promptValue = prompt.format(user_input="welcom to califonya!")
llm_output = llm(promptValue)

# Sử dụng trình phân tích kết xuất để phân tích nội dung được tạo ra
output_parser.parse(llm_output)
```

![image-20230406000017276](doc/image-20230406000017276.png)

### **Crawl trang web và xuất dữ liệu JSON**

Đôi khi chúng ta cần crawl một trang web có cấu trúc mạnh và trả về thông tin từ trang web đó dưới dạng JSON.

Chúng ta có thể sử dụng lớp `LLMRequestsChain` để thực hiện điều này, hãy tham khảo mã sau:

> Để dễ hiểu, trong ví dụ này tôi đã trực tiếp sử dụng phương thức của Prompt để định dạng kết quả đầu ra, thay vì sử dụng StructuredOutputParser như trong ví dụ trước, điều này cung cấp một cách tiếp cận khác để định dạng.

```python
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMRequestsChain, LLMChain

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

template = """Trong đoạn giữa >>> và <<< là nội dung HTML trả về từ trang web.
Trang web là mô tả vắn tắt của một công ty niêm yết trên thị trường chứng khoán A của Trang web Sina.
Hãy trích xuất thông tin từ trang web đã yêu cầu.

>>> {requests_result} <<<
Hãy trả về dữ liệu dưới định dạng JSON như sau
{{
  "company_name":"a",
  "company_english_name":"b",
  "issue_price":"c",
  "date_of_establishment":"d",
  "registered_capital":"e",
  "office_address":"f",
  "Company_profile":"g"

}}
Trích xuất:"""

prompt = PromptTemplate(
    input_variables=["requests_result"],
    template=template
)

chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))
inputs = {
  "url": "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/600519.phtml"
}

response = chain(inputs)
print(response['output'])
```

Chúng ta có thể thấy rằng nó đã xuất ra đúng kết quả đã được định dạng.

<figure><img src="doc/image-20230510234934.png" alt=""><figcaption></figcaption></figure>

### **Tùy chỉnh các công cụ được sử dụng trong agent**

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

llm = OpenAI(temperature=0)

# Khởi tạo chuỗi tìm kiếm và chuỗi tính toán
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

# Tạo danh sách các công cụ mà agent có sẵn, quá trình thực hiện của agent có thể được xem trong hình biểu đồ trong khái niệm quan trọng về agent
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="hữu ích khi bạn cần trả lời câu hỏi về các sự kiện hiện tại"
),
Tool(
name="Calculator",
func=llm_math_chain.run,
description="hữu ích khi bạn cần trả lời câu hỏi về toán học"
)
]
# Khởi tạo agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

#Thực hiện agent
agent.run("Bạn gái của Leo DiCaprio là ai? Tuổi hiện tại của cô ấy được gia hạn với mũi 0.43 là bao nhiêu?")
```

![image-20230406002117283](doc/image-20230406002117283.png)

Tự định nghĩa công cụ có một phần thú vị, sử dụng trọng số của công cụ dựa trên `nội dung mô tả trong công cụ`, hoàn toàn khác biệt so với việc lập trình trước đây dựa trên các giá trị số để kiểm soát trọng số.

Ví dụ, trong phần mô tả của Calculator, nói rằng nếu bạn có câu hỏi về toán học, hãy sử dụng công cụ này. Chúng ta có thể thấy trong quá trình thực thi ở trên, công cụ đã chọn Calculator để tính toán phần toán học trong prompt mà chúng ta yêu cầu.

### **Sử dụng Memory để tạo một trò chuyện bot với khả năng nhớ**

Trong ví dụ trước, chúng ta đã sử dụng cách lưu trữ lịch sử cuộc trò chuyện bằng cách tạo một danh sách tùy chỉnh.

Tất nhiên, bạn cũng có thể sử dụng đối tượng bộ nhớ tích hợp sẵn.

```python
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(temperature=0)

# Khởi tạo đối tượng MessageHistory
history = ChatMessageHistory()

# Thêm nội dung cuộc trò chuyện vào đối tượng MessageHistory
history.add_ai_message("Xin chào!")
history.add_user_message("Thủ đô của Trung Quốc là gì?")

# Thực thi cuộc trò chuyện
ai_response = chat(history.messages)
print(ai_response)
```

### **Sử dụng mô hình Hugging Face**
Trước khi sử dụng mô hình Hugging Face, bạn cần thiết lập biến môi trường trước

```python
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
```

Sử dụng mô hình Hugging Face trực tuyến

```python
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """Câu hỏi: {question}
Trả lời: Hãy suy nghĩ từng bước một."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":64})
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "Đội NFL nào đã giành chiến thắng trong Super Bowl trong năm Justin Beiber sinh ra?"
print(llm_chain.run(question))
```

Trích dẫn mô hình Hugging Face trực tiếp và sử dụng trên máy cục bộ

```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

model_id = 'google/flan-t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)

pipe = pipeline(
"text2text-generation",
model=model,
tokenizer=tokenizer,
max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
print(local_llm('Thủ đô của Pháp là gì? '))

llm_chain = LLMChain(prompt=prompt, llm=local_llm)
question = "Thủ đô của Anh là gì?"
print(llm_chain.run(question))
```


Lợi ích của việc trích dẫn mô hình và sử dụng trên máy cục bộ:

* Huấn luyện mô hình
* Có thể sử dụng GPU cục bộ
* Một số mô hình không thể chạy trên Hugging Face

### **Thực thi lệnh SQL bằng tự nhiên ngôn ngữ**

Chúng ta có thể thực thi lệnh SQL bằng `SQLDatabaseToolkit` hoặc `SQLDatabaseChain`

```python
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI

db = SQLDatabase.from_uri("sqlite:///../notebooks/Chinook.db")
toolkit = SQLDatabaseToolkit(db=db)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0),
    toolkit=toolkit,
    verbose=True
)

agent_executor.run("Mô tả bảng playlisttrack")
```

```python
from langchain import OpenAI, SQLDatabase, SQLDatabaseChain

db = SQLDatabase.from_uri("mysql+pymysql://root:root@127.0.0.1/chinook")
llm = OpenAI(temperature=0)

db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
db_chain.run("How many employees are there?")
```

## Tổng kết
Tất cả các ví dụ đã hoàn thành cơ bản, hy vọng các bạn có được những kiến thức từ bài viết này. Bài viết này chỉ là một giới thiệu sơ lược về LangChain, và tôi hy vọng rằng sẽ tiếp tục cập nhật chuỗi bài viết này nếu có những công nghệ tốt hơn sau này.

Vì LangChain đang phát triển rất nhanh, nên tôi tin chắc rằng nó sẽ tiếp tục phát triển các tính năng tốt hơn trong tương lai, do đó tôi rất tin tưởng vào thư viện mã nguồn mở này.

Hy vọng các bạn có thể phát triển các sản phẩm sáng tạo hơn bằng cách kết hợp với LangChain, thay vì chỉ tạo ra các sản phẩm chatbot tự động mà không có tính sáng tạo.

Các đoạn mã ví dụ trong bài viết này có sẵn ở đây, chúc các bạn học tập vui vẻ.

https://colab.research.google.com/drive/1ArRVMiS-YkhUlobHrU6BeS8fF57UeaPQ?usp=sharing
