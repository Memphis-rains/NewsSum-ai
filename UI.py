import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import traceback
import sys
from datetime import datetime


try:
    import torch
    TORCH_AVAILABLE = True
    torch_version = torch.__version__
except ImportError as e:
    TORCH_AVAILABLE = False
    torch_error = str(e)
    torch_version = "Not installed"

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    TRANSFORMERS_AVAILABLE = True
    import transformers
    transformers_version = transformers.__version__
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    transformers_error = str(e)
    transformers_version = "Not installed"

try:
    import numpy as np
    NUMPY_AVAILABLE = True
    numpy_version = np.__version__
except ImportError as e:
    NUMPY_AVAILABLE = False
    numpy_error = str(e)
    numpy_version = "Not installed"

# Page configuration
st.set_page_config(
    page_title="NewsSum.ai",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
            body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: white;
        color: #333;
        margin: 0;
        padding: 0;
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
            color: white;
        background:white;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .summary-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        font-size: 1.1rem;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .error-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffa726 0%, #ff7043 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class NewsumAI:
    def __init__(self, model_path="./finetuned_cnn_dm_sentiment/t5"):
        self.model_path = model_path
        self.device = None
        self.model = None
        self.tokenizer = None
        self.sentiment_pipelines = {}
        self.initialization_errors = []
        
        # Check device availability
        try:
            if TORCH_AVAILABLE:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                st.info(f"🔧 Device selected: {self.device}")
            else:
                self.device = "cpu"
                st.warning("⚠️ PyTorch not available, device set to CPU")
        except Exception as e:
            self.device = "cpu"
            self.initialization_errors.append(f"Device selection error: {str(e)}")
            st.error(f"❌ Device selection failed: {str(e)}")
    
    def load_model(self):
        """Load the fine-tuned model with comprehensive error handling"""
        if not TORCH_AVAILABLE:
            error_msg = f"PyTorch not available: {torch_error if 'torch_error' in globals() else 'Unknown error'}"
            st.error(f"❌ {error_msg}")
            return False, error_msg
            
        if not TRANSFORMERS_AVAILABLE:
            error_msg = f"Transformers not available: {transformers_error if 'transformers_error' in globals() else 'Unknown error'}"
            st.error(f"❌ {error_msg}")
            return False, error_msg
        
        try:
            st.info(f"🔍 Looking for model at: {self.model_path}")
            
            # Check if model directory exists
            import os
            if not os.path.exists(self.model_path):
                error_msg = f"Model directory not found: {self.model_path}"
                st.error(f"❌ {error_msg}")
                st.info("💡 Make sure your fine-tuned model is saved in the correct directory")
                return False, error_msg
            
            # List contents of model directory for debugging
            try:
                model_files = os.listdir(self.model_path)
                st.info(f"📁 Files in model directory: {model_files}")
            except Exception as e:
                st.warning(f"⚠️ Could not list model directory contents: {str(e)}")
            
            # Load tokenizer
            st.info("🔧 Loading tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
                st.success("✅ Tokenizer loaded successfully")
            except Exception as e:
                error_msg = f"Tokenizer loading failed: {str(e)}"
                st.error(f"❌ {error_msg}")
                st.code(f"Full traceback:\n{traceback.format_exc()}")
                return False, error_msg
            
            # Load model
            st.info("🔧 Loading model...")
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
                self.model = self.model.to(self.device)
                self.model.eval()
                st.success("✅ Model loaded successfully")
                st.info(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            except Exception as e:
                error_msg = f"Model loading failed: {str(e)}"
                st.error(f"❌ {error_msg}")
                st.code(f"Full traceback:\n{traceback.format_exc()}")
                return False, error_msg
            
            return True, "Model loaded successfully"
            
        except Exception as e:
            error_msg = f"Unexpected error in model loading: {str(e)}"
            st.error(f"❌ {error_msg}")
            st.code(f"Full traceback:\n{traceback.format_exc()}")
            return False, error_msg
    
    def setup_sentiment_pipelines(self):
        """Initialize sentiment analysis pipelines with detailed error handling"""
        if not TRANSFORMERS_AVAILABLE:
            error_msg = "Transformers not available for sentiment analysis"
            st.warning(f"⚠️ {error_msg}")
            return False, error_msg
        
        success_count = 0
        errors = []
        
        # Polarity analysis
        try:
            st.info("🔧 Loading polarity sentiment pipeline...")
            self.sentiment_pipelines['polarity'] = pipeline("sentiment-analysis")
            st.success("✅ Polarity sentiment pipeline loaded")
            success_count += 1
        except Exception as e:
            error_msg = f"Polarity pipeline failed: {str(e)}"
            errors.append(error_msg)
            st.error(f"❌ {error_msg}")
        
        # Emotion analysis
        try:
            st.info("🔧 Loading emotion analysis pipeline...")
            self.sentiment_pipelines['emotion'] = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base", 
                return_all_scores=True
            )
            st.success("✅ Emotion analysis pipeline loaded")
            success_count += 1
        except Exception as e:
            error_msg = f"Emotion pipeline failed: {str(e)}"
            errors.append(error_msg)
            st.error(f"❌ {error_msg}")
        
        # Zero-shot classification
        try:
            st.info("🔧 Loading zero-shot classification pipeline...")
            self.sentiment_pipelines['zero_shot'] = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli"
            )
            st.success("✅ Zero-shot classification pipeline loaded")
            success_count += 1
        except Exception as e:
            error_msg = f"Zero-shot pipeline failed: {str(e)}"
            errors.append(error_msg)
            st.error(f"❌ {error_msg}")
        
        if success_count > 0:
            return True, f"Loaded {success_count}/3 sentiment pipelines successfully"
        else:
            return False, f"All sentiment pipelines failed: {'; '.join(errors)}"
    
    def analyze_text(self, article_text):
        """Analyze text with comprehensive error handling"""
        if not self.model or not self.tokenizer:
            error_msg = "Model or tokenizer not loaded"
            st.error(f"❌ {error_msg}")
            return None
            
        try:
            st.info("🔧 Preparing input text...")
            
            # Generate summary
            prefix = "summarize: "
            try:
                inputs = self.tokenizer(
                    prefix + article_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                st.info("✅ Text tokenized successfully")
            except Exception as e:
                st.error(f"❌ Tokenization failed: {str(e)}")
                return None
            
            try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                st.info(f"✅ Inputs moved to device: {self.device}")
            except Exception as e:
                st.error(f"❌ Failed to move inputs to device: {str(e)}")
                return None
            
            try:
                st.info("🔧 Generating summary...")
                with torch.no_grad():
                    ids = self.model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                st.info("✅ Summary generated successfully")
            except Exception as e:
                st.error(f"❌ Summary generation failed: {str(e)}")
                st.code(f"Full traceback:\n{traceback.format_exc()}")
                return None
            
            try:
                summary = self.tokenizer.decode(ids[0], skip_special_tokens=True)
                st.info(f"✅ Summary decoded: {len(summary)} characters")
            except Exception as e:
                st.error(f"❌ Summary decoding failed: {str(e)}")
                return None
            
            # Prepare result
            result = {
                "summary": summary,
                "article_length": len(article_text.split()),
                "summary_length": len(summary.split()),
                "compression_ratio": len(summary.split()) / len(article_text.split()) * 100
            }
            
            # Sentiment analysis with individual error handling
            sentiment_errors = []
            
            # Polarity analysis
            if 'polarity' in self.sentiment_pipelines:
                try:
                    polarity = self.sentiment_pipelines['polarity'](summary)[0]
                    result["polarity"] = polarity
                    st.info("✅ Polarity analysis completed")
                except Exception as e:
                    sentiment_errors.append(f"Polarity analysis failed: {str(e)}")
                    st.warning(f"⚠️ Polarity analysis failed: {str(e)}")
            
            # Emotion analysis
            if 'emotion' in self.sentiment_pipelines:
                try:
                    emotions = self.sentiment_pipelines['emotion'](summary)[0]
                    result["emotions"] = emotions
                    st.info("✅ Emotion analysis completed")
                except Exception as e:
                    sentiment_errors.append(f"Emotion analysis failed: {str(e)}")
                    st.warning(f"⚠️ Emotion analysis failed: {str(e)}")
            
            # Intent and aspect classification
            if 'zero_shot' in self.sentiment_pipelines:
                try:
                    intent_labels = ["informative", "warning", "opinion", "breaking news", "analysis"]
                    aspect_labels = ["economy", "politics", "health", "sports", "technology", "entertainment"]
                    
                    intent = self.sentiment_pipelines['zero_shot'](summary, candidate_labels=intent_labels)
                    aspects = self.sentiment_pipelines['zero_shot'](summary, candidate_labels=aspect_labels)
                    
                    result["intent"] = intent
                    result["aspects"] = aspects
                    st.info("✅ Classification analysis completed")
                except Exception as e:
                    sentiment_errors.append(f"Classification failed: {str(e)}")
                    st.warning(f"⚠️ Classification failed: {str(e)}")
            
            if sentiment_errors:
                result["sentiment_errors"] = sentiment_errors
            
            return result
            
        except Exception as e:
            st.error(f"❌ Unexpected error in text analysis: {str(e)}")
            st.code(f"Full traceback:\n{traceback.format_exc()}")
            return None

def create_emotion_chart(emotions_data):
    """Create an interactive emotion analysis chart"""
    if not emotions_data:
        return None
        
    try:
        df = pd.DataFrame(emotions_data)
        df = df.sort_values('score', ascending=True)
        
        fig = px.bar(
            df, 
            x='score', 
            y='label',
            orientation='h',
            color='score',
            color_continuous_scale='viridis',
            title="Emotion Analysis",
            labels={'score': 'Confidence Score', 'label': 'Emotion'}
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating emotion chart: {str(e)}")
        return None

def create_classification_chart(data, title):
    """Create classification charts for intent/aspects"""
    if not data or not data.get('labels'):
        return None
        
    try:
        df = pd.DataFrame({
            'labels': data['labels'][:5],  # Top 5
            'scores': data['scores'][:5]
        })
        
        fig = go.Figure(data=[
            go.Bar(
                x=df['labels'],
                y=df['scores'],
                marker_color=px.colors.qualitative.Set3[:len(df)],
                text=[f'{score:.2f}' for score in df['scores']],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Category",
            yaxis_title="Confidence Score",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating classification chart: {str(e)}")
        return None

def create_metrics_overview(result):
    """Create metrics overview cards"""
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📝 Words</h3>
                <h2>{result['article_length']}</h2>
                <p>Original Article</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>✂️ Summary</h3>
                <h2>{result['summary_length']}</h2>
                <p>Words Generated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            compression = f"{result['compression_ratio']:.1f}%"
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Compression</h3>
                <h2>{compression}</h2>
                <p>Reduction Ratio</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            sentiment_label = result.get('polarity', {}).get('label', 'N/A')
            sentiment_color = {'POSITIVE': '🟢', 'NEGATIVE': '🔴', 'NEUTRAL': '🟡'}.get(sentiment_label, '⚪')
            st.markdown(f"""
            <div class="metric-card">
                <h3>💭 Sentiment</h3>
                <h2>{sentiment_color}</h2>
                <p>{sentiment_label}</p>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error creating metrics overview: {str(e)}")

def display_system_info():
    """Display detailed system information for debugging"""
    with st.expander("🔍 System Information & Debugging"):
        st.markdown("### Python Environment")
        st.info(f"Python version: {sys.version}")
        
        st.markdown("### Package Versions")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**PyTorch:** {torch_version}")
            if not TORCH_AVAILABLE:
                st.error(f"Error: {torch_error if 'torch_error' in globals() else 'Unknown'}")
            
            st.markdown(f"**Transformers:** {transformers_version}")
            if not TRANSFORMERS_AVAILABLE:
                st.error(f"Error: {transformers_error if 'transformers_error' in globals() else 'Unknown'}")
        
        with col2:
            st.markdown(f"**NumPy:** {numpy_version}")
            if not NUMPY_AVAILABLE:
                st.error(f"Error: {numpy_error if 'numpy_error' in globals() else 'Unknown'}")
            
            # Check CUDA availability
            if TORCH_AVAILABLE:
                cuda_available = torch.cuda.is_available()
                st.markdown(f"**CUDA Available:** {'Yes' if cuda_available else 'No'}")
                if cuda_available:
                    st.markdown(f"**CUDA Version:** {torch.version.cuda}")
                    st.markdown(f"**GPU Count:** {torch.cuda.device_count()}")
        
        st.markdown("### Installation Commands")
        st.code("""
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install transformers datasets evaluate streamlit plotly pandas numpy

# For GPU version (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        """)

def main():
    # Header
    st.markdown('<h1 class="main-header">📰 NewsSum.ai</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced AI-Powered News Summarization & Sentiment Analysis Platform</p>', unsafe_allow_html=True)
    
    # Display system info for debugging
    display_system_info()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.markdown("## 🎯 About NewsSum.ai")
        st.markdown("""
        **NewsSum.ai** leverages cutting-edge transformer technology to:
        
        • **Summarize** lengthy news articles instantly
        • **Analyze** sentiment and emotional tone
        • **Classify** content by intent and topic
        • **Visualize** insights through interactive charts
        
        Built with fine-tuned T5 models and advanced NLP pipelines.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model status
        st.markdown("## 🔧 System Status")
        
        # Check dependencies first
        if not TORCH_AVAILABLE:
            st.markdown("""
            <div class="error-card">
                <h4>❌ PyTorch Missing</h4>
                <p>Please install PyTorch to continue</p>
            </div>
            """, unsafe_allow_html=True)
        
        if not TRANSFORMERS_AVAILABLE:
            st.markdown("""
            <div class="error-card">
                <h4>❌ Transformers Missing</h4>
                <p>Please install transformers library</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Initialize the model only if dependencies are available
        if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
            newssum = NewsumAI()
            
            with st.spinner("Loading AI models..."):
                model_loaded, model_message = newssum.load_model()
                sentiment_loaded, sentiment_message = newssum.setup_sentiment_pipelines()
            
            if model_loaded:
                st.success("✅ Summarization Model Ready")
            else:
                st.error(f"❌ Model Loading Failed: {model_message}")
                
            if sentiment_loaded:
                st.success("✅ Sentiment Analysis Ready")
            else:
                st.warning(f"⚠️ Sentiment Features Limited: {sentiment_message}")
        else:
            model_loaded = False
            newssum = None
        
        st.markdown("---")
        
        # Sample articles
        st.markdown("## 📄 Sample Articles")
        sample_articles = {
    "Informative News": """By mid-August 2025, solar power generation in Britain has already exceeded the total output for all of 2024, marking a pivotal moment in the UK’s renewable energy trajectory. According to combined data from the Financial Times and Sheffield University, around 14.08 TWh of electricity has been generated so far this year—representing a 33 % year-on-year increase and enough energy to power 5.2 million homes.

This surge has been fueled by two main factors: a 20 % increase in solar capacity since 2023 and record levels of sunshine earlier this year. Solar energy now supplies nearly 10 % of electricity needs across England and Wales from January through July.

The UK government continues to pursue its ambitious target of 45–47 GW of solar capacity by 2030, supported by policies including mandatory rooftop panels on new homes, financial incentives, and relaxed planning regulations. Yet, as the pace accelerates, challenges loom: existing grid infrastructure is under pressure, local resistance to new solar farms is growing, and only one large-scale solar-plus-storage project, Cleve Hill, is currently operational. To meet demand, around 80 additional large projects are needed.

The government has greenlighted £10 billion for grid upgrades, indicating substantial investment to integrate more renewables into the national system. Developers are increasingly incorporating battery storage into solar farms, aiming to mitigate peak demand and ensure energy stability.

This boom in solar energy production is not just environmentally ambitious; it reveals a strategic shift toward energy independence and resilience. But balancing growth with infrastructure planning, community buy-in, and storage solutions remains crucial if the UK is to meet its net-zero commitments effectively.

(Source: Financial Times)""",

    "Warning News": """In a significant public safety initiative, the UK government announced that on Sunday, 7 September 2025 at approximately 3 pm, an emergency alert test will be conducted across Scotland. The test will utilize mobile phones on 4G and 5G networks, emitting a loud siren-like sound for up to 10 seconds, accompanied by a message clearly stating it is a test and that no action is needed.

This large-scale drill aims to verify the reliability of the emergency alert system and familiarise the public with its functionality. It’s one of the most significant safety exercises ever undertaken in the UK. The alert will include a link to government-issued emergency preparedness advice, offering guidance on how to respond should a real crisis occur.

This system played a vital role earlier this year during Storm Eowyn in January 2025, when 4.5 million people across the UK were notified to stay indoors via emergency alerts amid a red weather warning. Preparing for extreme weather such as floods or storms, and even civil emergencies, this test aims to ensure readiness across both the public and agencies.

Emergency services and local authorities are gearing up to support the system; however, residents are encouraged not to alarm when their phone resonates loudly—but to recognise it as a routine, necessary safety check.

(Source: The Scottish Sun)""",

    "Analysis News": """By mid-August 2025, Britain’s solar power generation already surpasses the total recorded in 2024, underscoring a sharp acceleration in renewable energy output. Around 14.08 TWh of electricity has been generated so far this year—representing a 33 % increase year-on-year, enough to power more than 5 million homes.

The remarkable surge is primarily due to two drivers: a 20 % increase in installed solar capacity since 2023 and record levels of sunshine earlier this year. Solar now supplies nearly 10 % of England and Wales’ electricity demand, making it one of the fastest-growing segments of the energy mix.

Analysts note that while this is a strong sign of progress, challenges remain. The UK must upgrade grid infrastructure, expand storage capacity, and address community concerns around large-scale solar farms. Only one industrial-scale solar-plus-storage site, Cleve Hill, is currently operational; to meet rising demand, around 80 more projects of similar size are needed.

The government has earmarked £10 billion for grid improvements, reflecting a commitment to integrate renewables on a wider scale. Developers are also investing in battery solutions to help smooth supply during peaks and troughs in demand. Without such upgrades, bottlenecks in energy distribution may limit the benefits of the solar surge.

Ultimately, the story of Britain’s solar growth is about more than numbers—it’s about energy independence, climate responsibility, and the balance between environmental goals and practical limitations. The trend points to a brighter energy future, provided the momentum is paired with sustainable infrastructure planning.

(Source: Financial Times)""",

    "Opinion News": """In a recent letter to the Financial Times, writer John Murray argued that the ongoing debate about child poverty should focus more squarely on parental responsibility than state welfare. While he acknowledged that children must never be punished for circumstances beyond their control, he emphasized that families—and parents in particular—should bear the primary responsibility for ensuring a stable upbringing.

Murray expressed concern that public debates often overemphasize government aid, while underplaying the role of individual decisions such as family planning and financial preparedness. In his view, the long-term solution lies in education and awareness: equipping prospective parents with the knowledge and tools to raise children responsibly, rather than depending excessively on welfare systems.

He noted the growing societal trend of delayed or avoided parenthood due to economic constraints, but warned that those who do choose to have children must be fully prepared—emotionally, financially, and logistically. While welfare systems and taxation policies remain crucial for immediate relief, Murray contends that overreliance on these tools risks masking deeper social challenges.

The letter concludes by urging policymakers and communities alike to reconsider the balance between state support and family responsibility. Although controversial, the opinion reflects a growing perspective in public discourse—that lasting solutions to poverty require a cultural shift in how families prepare for the responsibilities of parenthood.

(Source: Financial Times)""",

    "Breaking News": """Ministers have announced the introduction of a fast-track processing system for asylum applications, following mounting public unrest and protests near hotels currently housing asylum seekers. As of Sunday, 24 August 2025, senior government figures confirmed the new mechanism aims to streamline case reviews, reducing delays that have fueled political pressure and community tensions.

The announcement comes amid a wave of demonstrations across the UK. Many locals voiced frustration about disruptions around hotels used to house asylum seekers, prompting authorities to intervene and maintain public order. The protests, particularly intense in affected communities, highlighted widespread dissatisfaction with the pace and transparency of the asylum system.

Government spokespeople insist the fast-track approach will be comprehensive, prioritising vulnerable individuals, such as children and those facing potential harm. The initiative is designed to expedite decisions without compromising the integrity of the process. While exact timelines weren’t detailed, the move signals an urgent response intended to restore public confidence in managing asylum flows more effectively.

Opposition parties have mounted early criticism, suggesting that without proper safeguards, the fast-track system may compromise human rights standards. Campaigners warn against rushing through decisions that require careful consideration and due process.

As this story develops, updates are expected regarding specific procedural changes, impact assessments, and how the new system integrates with existing infrastructure.

(Source: Sky News)"""
}

        
        selected_sample = st.selectbox("Choose a sample:", ["Custom"] + list(sample_articles.keys()))
    
    # Main content area
    if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        st.markdown("""
        <div class="error-card">
            <h3>⚠️ Missing Dependencies</h3>
            <p>Please install the required packages to use NewsSum.ai</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if not model_loaded:
        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ Model Not Available</h3>
            <p>Please ensure your fine-tuned model is available in the correct directory.</p>
            <p><strong>Expected path:</strong> <code>./finetuned_cnn_dm_sentiment/t5/</code></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Input section
    st.markdown("## 📝 Input Your News Article")
    
    # Use sample or custom input
    if selected_sample != "Custom":
        default_text = sample_articles[selected_sample]
    else:
        default_text = ""
    
    article_text = st.text_area(
        "Paste your news article here:",
        value=default_text,
        height=200,
        placeholder="Enter a news article for analysis... >200 words recommended"
    )
    
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🚀 Analyze Article", use_container_width=True)
    
    # Analysis results
    if analyze_button and article_text.strip():
        with st.spinner("🤖 AI is analyzing your article..."):
            # Show progress bar
            progress_bar = st.progress(0)
            for i in range(50):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            # Perform analysis
            result = newssum.analyze_text(article_text)
            
            # Complete progress bar
            for i in range(50, 100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            progress_bar.empty()
        
        if result:
            st.success("✅ Analysis Complete!")
            
            # Show any sentiment errors
            if 'sentiment_errors' in result:
                with st.expander("⚠️ Partial Analysis Warnings"):
                    for error in result['sentiment_errors']:
                        st.warning(error)
            
            # Metrics overview
            st.markdown("## 📊 Overview Metrics")
            create_metrics_overview(result)
            
            # Summary section
            st.markdown("## 📄 AI-Generated Summary")
            st.markdown(f"""
            <div class="summary-box">
                <h4>Summary:</h4>
                <p>{result['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Analysis charts
            st.markdown("## 📈 Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Emotion analysis
                if 'emotions' in result:
                    emotion_fig = create_emotion_chart(result['emotions'])
                    if emotion_fig:
                        st.plotly_chart(emotion_fig, use_container_width=True)
                
                # Intent classification
                if 'intent' in result:
                    intent_fig = create_classification_chart(result['intent'], "Intent Classification")
                    if intent_fig:
                        st.plotly_chart(intent_fig, use_container_width=True)
            
            with col2:
                # Sentiment gauge
                if 'polarity' in result:
                    sentiment_score = result['polarity']['score']
                    sentiment_label = result['polarity']['label']
                    
                    try:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = sentiment_score,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': f"Sentiment: {sentiment_label}"},
                            gauge = {
                                'axis': {'range': [None, 1]},
                                'bar': {'color': "darkgreen" if sentiment_label == "POSITIVE" else "darkred"},
                                'steps': [
                                    {'range': [0, 0.3], 'color': "lightgray"},
                                    {'range': [0.3, 0.7], 'color': "gray"},
                                    {'range': [0.7, 1], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 0.9
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Error creating sentiment gauge: {str(e)}")
                
                # Aspect classification
                if 'aspects' in result:
                    aspect_fig = create_classification_chart(result['aspects'], "Topic Classification")
                    if aspect_fig:
                        st.plotly_chart(aspect_fig, use_container_width=True)
            
            # Detailed results
            with st.expander("🔍 View Detailed Results"):
                st.json(result)
            
            # Export options
            st.markdown("## 💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    # Download JSON
                    json_str = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download JSON Report",
                        data=json_str,
                        file_name=f"newssum_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"❌ Error creating JSON download: {str(e)}")
            
            with col2:
                try:
                    # Download CSV
                    df = pd.DataFrame([result])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV Report",
                        data=csv,
                        file_name=f"newssum_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"❌ Error creating CSV download: {str(e)}")
        
        else:
            st.error("❌ Analysis failed. Please check the error messages above and try again.")
    
    elif analyze_button and not article_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>NewsSum.ai | Powered by Advanced AI & Machine Learning</p>
        <p>Built with Streamlit, PyTorch, Transformers & Plotly</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.code(f"Full traceback:\n{traceback.format_exc()}")
        st.info("Please check the error details above and ensure all dependencies are installed correctly.")