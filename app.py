import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
import tempfile
import os
import json

# Streamlit App
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

# Get the Groq API Key and url(YT or website)to be summarized
with st.sidebar:
    google_api_key = st.text_input("Google API Key", value="", type="password")
    show_transcript = st.checkbox("Show extracted transcript", value=False)

generic_url = st.text_input("URL", label_visibility="collapsed")

def is_youtube_url(url):
    """Check if URL is from YouTube"""
    return "youtube.com" in url or "youtu.be" in url

def extract_youtube_transcript(video_url):
    """Extract transcript using yt-dlp"""
    try:
        import yt_dlp
    except ImportError:
        st.error("yt-dlp is not installed. Please run: pip install yt-dlp")
        st.stop()
    
    # Create a temporary directory for any files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up yt-dlp options to extract subtitles
        options = {
            'writesubtitles': True,
            'writeautomaticsub': True,  # Get auto-generated if no manual subs
            'subtitleslangs': ['en', 'en-US', 'en-GB'],  # Try English first
            'skip_download': True,  # Don't download the video
            'outtmpl': os.path.join(temp_dir, 'subtitle'),  # Output template
            'quiet': True,
            'no_warnings': True
        }
        
        # Try to extract info and subtitles
        with yt_dlp.YoutubeDL(options) as ydl:
            st.info("Extracting video information...")
            try:
                info = ydl.extract_info(video_url, download=False)
                
                # Get video title
                video_title = info.get('title', 'YouTube Video')
                
                # Check if subtitles were extracted
                if 'requested subtitles' in info and info['requested subtitles']:
                    subtitle_data = info['requested subtitles']
                    
                    for lang, subtitle_info in subtitle_data.items():
                        # Get subtitle format
                        if subtitle_info.get('data'):
                            # Subtitle data is directly available
                            subtitle_content = subtitle_info['data']
                            
                            # Process based on format (vtt, srt, etc.)
                            if subtitle_info.get('ext') == 'vtt':
                                # Process WebVTT format
                                import re
                                # Remove timestamp lines and format tags
                                cleaned_lines = []
                                for line in subtitle_content.split('\n'):
                                    # Skip timing, style and empty lines
                                    if re.match(r'^\d{2}:\d{2}:\d{2}', line) or line.startswith('WEBVTT') or line.strip() == '' or '-->' in line:
                                        continue
                                    # Remove HTML-like tags
                                    line = re.sub(r'<[^>]+>', '', line)
                                    cleaned_lines.append(line)
                                
                                full_transcript = ' '.join(cleaned_lines)
                                return full_transcript, video_title
                            
                            elif subtitle_info.get('ext') == 'json':
                                # Process JSON format
                                try:
                                    json_data = json.loads(subtitle_content)
                                    # Extract text from JSON (format may vary)
                                    if isinstance(json_data, list):
                                        # Try common formats
                                        if all(isinstance(item, dict) for item in json_data):
                                            if all('text' in item for item in json_data):
                                                texts = [item['text'] for item in json_data]
                                                return ' '.join(texts), video_title
                                except:
                                    st.warning("Could not parse JSON subtitle format")
                            
                            else:
                                # Generic handling - just extract text without timestamps
                                import re
                                # Remove timestamps and formatting
                                lines = subtitle_content.split('\n')
                                cleaned_lines = []
                                for line in lines:
                                    # Skip likely timestamp lines
                                    if re.match(r'^\d', line) or '-->' in line or line.strip() == '':
                                        continue
                                    # Remove basic formatting
                                    line = re.sub(r'<[^>]+>', '', line)
                                    cleaned_lines.append(line)
                                
                                full_transcript = ' '.join(cleaned_lines)
                                return full_transcript, video_title
                
                # If we couldn't get subtitles directly, try the automatic captions
                if 'automatic_captions' in info and info['automatic_captions']:
                    auto_captions = info['automatic_captions']
                    for lang in ['en', 'en-US', 'en-GB']:
                        if lang in auto_captions:
                            caption_formats = auto_captions[lang]
                            # Try to find a text-based format
                            for fmt in caption_formats:
                                if fmt.get('ext') in ['vtt', 'srt', 'json']:
                                    try:
                                        with yt_dlp.YoutubeDL({
                                            'writeautomaticsub': True,
                                            'subtitleslangs': [lang],
                                            'subtitlesformat': fmt['ext'],
                                            'skip_download': True,
                                            'outtmpl': os.path.join(temp_dir, 'auto'),
                                            'quiet': True
                                        }) as auto_ydl:
                                            auto_ydl.download([video_url])
                                            
                                            # Find the subtitle file
                                            for file in os.listdir(temp_dir):
                                                if file.endswith(fmt['ext']):
                                                    with open(os.path.join(temp_dir, file), 'r', encoding='utf-8') as f:
                                                        subtitle_content = f.read()
                                                        
                                                    # Process subtitle content based on format
                                                    import re
                                                    cleaned_lines = []
                                                    for line in subtitle_content.split('\n'):
                                                        # Skip timestamp lines and empty lines
                                                        if re.match(r'^\d', line) or '-->' in line or line.strip() == '':
                                                            continue
                                                        # Remove HTML-like tags
                                                        line = re.sub(r'<[^>]+>', '', line)
                                                        cleaned_lines.append(line)
                                                    
                                                    full_transcript = ' '.join(cleaned_lines)
                                                    return full_transcript, video_title
                                    except Exception as e:
                                        st.warning(f"Error extracting automatic captions: {str(e)}")
                
                # Try fallback method with description
                if 'description' in info and info['description']:
                    st.warning("No transcript found. Using video description as fallback.")
                    return info['description'], video_title
                
                # Last resort - use title
                st.error("Could not extract any text content from this video.")
                return f"Video title: {video_title}. No transcript available.", video_title
                
            except Exception as e:
                st.error(f"Error extracting video information: {str(e)}")
                return None, None
    
    return None, None

# Initialize Groq LLM
def get_llm(api_key):
    """Initialize the LLM with error handling"""
    try:
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {str(e)}")
        st.stop()

# Summarization prompt
prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    # Validate all the inputs
    if not google_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL")
    else:
        try:
            # Initialize the LLM early to catch API key issues
            llm = get_llm(google_api_key)
            
            with st.spinner("Loading content..."):
                # Check if it's a YouTube URL
                if is_youtube_url(generic_url):
                    st.info("Processing YouTube video...")
                    
                    # Extract transcript using yt-dlp
                    transcript, video_title = extract_youtube_transcript(generic_url)
                    
                    if transcript:
                        st.success(f"Successfully extracted content from: {video_title}")
                        
                        if show_transcript:
                            with st.expander("Extracted Content"):
                                st.write(transcript[:1000] + "..." if len(transcript) > 1000 else transcript)
                        
                        # Create a Document object for the summarization chain
                        docs = [Document(page_content=transcript, metadata={"source": generic_url, "title": video_title})]
                    else:
                        st.error("Could not extract content from this YouTube video.")
                        st.stop()
                else:
                    # Handle regular website
                    st.info(f"Processing website: {generic_url}")
                    
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs = loader.load()
                    
                    if not docs or len(docs) == 0 or not docs[0].page_content.strip():
                        st.error("Could not extract content from the website.")
                        st.stop()
                    
                    st.success("Successfully extracted content from website")
                    
                    if show_transcript:
                        with st.expander("Extracted Content"):
                            st.write(docs[0].page_content[:1000] + "..." if len(docs[0].page_content) > 1000 else docs[0].page_content)
            
            # Summarization
            with st.spinner("Generating summary..."):
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)
                
                st.subheader("Summary")
                st.success(output_summary)
                
        except Exception as e:
            import traceback
            st.error("An error occurred during processing.")
            st.code(traceback.format_exc(), language="python")
            st.exception(f"Exception: {e}")
