"""
Talk-to-Data Chatbot using Groq API
Converts natural language questions to SQL queries and returns plain English answers
"""

import pandas as pd
import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()


class CreditRiskChatbot:
    """Chatbot for querying credit risk data using natural language"""
    
    def __init__(self):
        """Initialize the chatbot with database and Groq API"""
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.client = Groq(api_key=self.api_key)
        self.db_path = None
        self.setup_database()
    
    def setup_database(self):
        """Load data into SQLite database"""
        print("Setting up SQLite database...")
        
        # Load the training data
        data_path = Path(__file__).parent.parent / "data" / "application_train.csv"
        df = pd.read_csv(data_path)
        
        # Create database in data directory
        db_dir = Path(__file__).parent.parent / "data"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / "credit_risk.db"
        
        # Connect to database and load data
        conn = sqlite3.connect(self.db_path)
        df.to_sql('applications', conn, if_exists='replace', index=False)
        conn.close()
        
        print(f"✓ Database created at: {self.db_path}")
        print(f"✓ Loaded {len(df):,} records into 'applications' table")
    
    def get_table_schema(self):
        """Get the schema of the applications table"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(applications)")
        columns = cursor.fetchall()
        conn.close()
        
        schema = "Table: applications\nColumns:\n"
        for col in columns:
            schema += f"  - {col[1]} ({col[2]})\n"
        
        return schema
    
    def natural_language_to_sql(self, question):
        """Convert natural language question to SQL using Groq API"""
        schema = self.get_table_schema()
        
        prompt = f"""You are a SQL expert. Convert the following natural language question into a SQL query.

Database Schema:
{schema}

Important notes:
- The table name is 'applications'
- TARGET column: 0 = Non-Default, 1 = Default
- CODE_GENDER: M = Male, F = Female
- AMT_INCOME_TOTAL: Total income of the applicant
- AMT_CREDIT: Loan amount
- FLAG_OWN_CAR: Y = Owns car, N = Does not own car
- NAME_EDUCATION_TYPE: Education level of the applicant
- DAYS_BIRTH: Age in days (negative value, convert to years by dividing by -365.25)

Question: {question}

Return ONLY the SQL query, nothing else. Do not include any explanation or markdown formatting."""
        
        try:
            message = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            
            sql_query = message.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if sql_query.startswith("```"):
                sql_query = sql_query.split("```")[1]
                if sql_query.startswith("sql"):
                    sql_query = sql_query[3:]
                sql_query = sql_query.strip()
            
            return sql_query
        except Exception as e:
            raise Exception(f"Error converting natural language to SQL: {str(e)}")
    
    def execute_sql_query(self, sql_query):
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            conn.close()
            
            return results, columns
        except Exception as e:
            raise Exception(f"Error executing SQL query: {str(e)}")
    
    def format_answer(self, question, sql_query, results, columns):
        """Format the SQL results into a plain English answer using Groq API"""
        if not results:
            return "No results found for your query."
        
        # Convert results to a readable format
        results_str = "\n".join([str(row) for row in results])
        
        prompt = f"""You are a helpful data analyst. Convert the following SQL query results into a plain English answer to the user's question.

Original Question: {question}

SQL Query: {sql_query}

Results:
{results_str}

Columns: {', '.join(columns)}

Provide a clear, conversational answer. Include specific numbers and percentages where relevant. Be concise but informative."""
        
        try:
            message = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            
            answer = message.choices[0].message.content.strip()
            return answer
        except Exception as e:
            # Fallback to simple formatting if Groq fails
            return f"Query executed successfully. Results: {results_str}"
    
    def ask(self, question):
        """Process a natural language question and return the answer"""
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        try:
            # Convert to SQL
            print("Converting to SQL...")
            sql_query = self.natural_language_to_sql(question)
            print(f"SQL Query: {sql_query}")
            
            # Execute query
            print("Executing query...")
            results, columns = self.execute_sql_query(sql_query)
            print(f"Results: {len(results)} row(s) returned")
            
            # Format answer
            print("Formatting answer...")
            answer = self.format_answer(question, sql_query, results, columns)
            
            print(f"\nAnswer: {answer}")
            return answer
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(f"\nError: {error_msg}")
            return error_msg
    
    def close(self):
        """Close database connection"""
        print("Database connection closed")


def test_predefined_questions(chatbot):
    """Test the chatbot with predefined questions"""
    print("\n" + "="*80)
    print("TESTING PREDEFINED QUESTIONS")
    print("="*80)
    
    questions = [
        "How many people defaulted?",
        "What is the average income of defaulters?",
        "Which gender has higher default rate?",
        "What is the average loan amount?",
        "How many applicants own a car?"
    ]
    
    for question in questions:
        chatbot.ask(question)
        print()


def interactive_mode(chatbot):
    """Run chatbot in interactive mode"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Type your questions about the credit risk data.")
    print("Type 'quit' or 'exit' to stop.")
    print("-"*80)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            chatbot.ask(question)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Main function to run the chatbot"""
    print("="*80)
    print("TALK-TO-DATA CHATBOT")
    print("="*80)
    
    try:
        # Initialize chatbot
        chatbot = CreditRiskChatbot()
        
        # Test predefined questions
        test_predefined_questions(chatbot)
        
        # Ask if user wants interactive mode
        print("\n" + "="*80)
        choice = input("Would you like to ask more questions interactively? (y/n): ").strip().lower()
        
        if choice in ['y', 'yes']:
            interactive_mode(chatbot)
        
        # Close connection
        chatbot.close()
        
        print("\n" + "="*80)
        print("CHATBOT SESSION ENDED")
        print("="*80)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
