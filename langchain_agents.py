"""
Three-Agent LangChain Pipeline for Patient Load Forecasting

Agent 1: Data Interpreter & ML Orchestrator
    - Fetches user environmental data
    - Calls ML prediction endpoint
    - Loads and filters hospitals by pincode

Agent 2: Medical Recommendation Generator
    - Analyzes prediction and hospital capabilities
    - Maps disease to required specialty
    - Ranks hospitals and generates recommendations

Agent 3: Feasibility & Practicality Validator
    - Validates recommendations against constraints
    - Filters unrealistic suggestions
    - Formats final UI-ready output
"""

import os
import json
import requests
import pandas as pd
from typing import Dict, List, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ===========================
# Initialize OpenRouter LLM
# ===========================
try:
    llm = ChatOpenAI(
        model="x-ai/grok-4.1-fast:free",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
    )
except Exception as e:
    print(f"[WARNING] Could not initialize LLM: {e}")
    llm = None

# ===========================
# AGENT 1: Data Interpreter & ML Orchestrator
# ===========================

def fetch_user_data_impl() -> str:
    """Fetches current user environmental data from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/user_data")
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        # Return mock data when backend is unavailable
        print(f"[WARNING] Backend not available ({str(e)}), using mock data")
        mock_data = {
            "pincode": 500001,
            "city": "Hyderabad",
            "aqi_index": 145.0,
            "temperature_mean_c": 28.5,
            "relative_humidity_mean": 68.0,
            "rain_mm": 3.5,
            "uv_index_mean": 7.2,
            "timestamp": "2024-11-29T12:00:00",
            "location": {
                "latitude": 17.385044,
                "longitude": 78.486671
            }
        }
        return json.dumps(mock_data, indent=2)

def call_ml_prediction_impl(params: str) -> str:
    """
    Calls ML prediction endpoint with environmental parameters.
    Input should be JSON string with: pincode, aqi_index, temperature_mean_c, 
    relative_humidity_mean, rain_mm, uv_index_mean
    """
    try:
        params_dict = json.loads(params)
        response = requests.post(
            f"{BACKEND_URL}/predict_forecast",
            json=params_dict,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        # Return mock prediction when backend is unavailable
        print(f"[WARNING] ML prediction endpoint not available ({str(e)}), using mock prediction")
        mock_prediction = {
            "forecast_next_3_days": {
                "day_1": {"patient_load": 145, "dominant_disease": "Respiratory Issues", "confidence": 0.82},
                "day_2": {"patient_load": 152, "dominant_disease": "Heat Stroke", "confidence": 0.78},
                "day_3": {"patient_load": 138, "dominant_disease": "Allergies", "confidence": 0.85}
            },
            "environmental_factors": params_dict,
            "prediction_timestamp": "2024-11-29T12:00:00"
        }
        return json.dumps(mock_prediction, indent=2)

def load_hospitals_by_pincode_impl(pincode: str) -> str:
    """
    Loads hospitals from CSV and filters by pincode.
    Returns hospitals matching the given pincode area.
    """
    try:
        pincode_int = int(pincode)
        df = pd.read_csv("hospital_details.csv")
        
        # Filter by exact pincode or nearby (within 10 pincode range)
        filtered = df[
            (df['pincode'] >= pincode_int - 10) & 
            (df['pincode'] <= pincode_int + 10)
        ]
        
        # Convert to dict for better readability
        hospitals = filtered.to_dict('records')
        
        # Limit to top 20 hospitals
        hospitals = hospitals[:20]
        
        return json.dumps({
            "total_hospitals_found": len(filtered),
            "hospitals_returned": len(hospitals),
            "hospitals": hospitals
        }, indent=2, default=str)
    except Exception as e:
        return f"Error loading hospitals: {str(e)}"

# Tool-wrapped versions for LangChain agents (simplified - no decorators needed)
def fetch_user_data() -> str:
    """Fetches current user environmental data from backend"""
    return fetch_user_data_impl()

def call_ml_prediction(params: str) -> str:
    """
    Calls ML prediction endpoint with environmental parameters.
    Input should be JSON string with: pincode, aqi_index, temperature_mean_c, 
    relative_humidity_mean, rain_mm, uv_index_mean
    """
    return call_ml_prediction_impl(params)

def load_hospitals_by_pincode(pincode: str) -> str:
    """
    Loads hospitals from CSV and filters by pincode.
    Returns hospitals matching the given pincode area.
    """
    return load_hospitals_by_pincode_impl(pincode)


def create_agent_1():
    """Creates Agent 1: Data Interpreter & ML Orchestrator"""
    
    tools = [fetch_user_data, call_ml_prediction, load_hospitals_by_pincode]
    
    return tools


def run_agent_1() -> Dict[str, Any]:
    """
    Executes Agent 1 workflow without LangChain agent (direct tool execution)
    Returns structured output for Agent 2
    """
    print("\n" + "="*60)
    print("[AGENT 1] Data Interpreter & ML Orchestrator")
    print("="*60)
    
    # Step 1: Fetch user data
    print("\n[Step 1] Fetching user environmental data...")
    user_data_str = fetch_user_data_impl()
    
    try:
        user_data = json.loads(user_data_str)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse user data JSON: {e}")
        print(f"Raw response: {user_data_str}")
        raise
    
    print(f"User Location: Pincode {user_data['pincode']}, {user_data['city']}")
    print(f"AQI: {user_data['aqi_index']}, Temp: {user_data['temperature_mean_c']}°C")

    # Step 2: Call ML prediction
    print("\n[Step 2] Calling ML prediction model...")
    ml_params = {
        "pincode": user_data["pincode"],
        "aqi_index": user_data["aqi_index"],
        "temperature_mean_c": user_data["temperature_mean_c"],
        "relative_humidity_mean": user_data["relative_humidity_mean"],
        "rain_mm": user_data["rain_mm"],
        "uv_index_mean": user_data["uv_index_mean"]
    }
    prediction_str = call_ml_prediction_impl(json.dumps(ml_params))
    prediction = json.loads(prediction_str)
    
    # Handle both old and new prediction format
    if 'predictions' in prediction:
        print(f"Predicted Disease Spike: {prediction['predictions'][0]['predicted_disease_spike']}")
        print(f"Intensity: {prediction['predictions'][0]['predicted_intensity']}")
        print(f"Dominant Risk: {prediction['predictions'][0]['dominant_risk']}")
    else:
        # New mock format
        forecast = prediction['forecast_next_3_days']
        print(f"Day 1 - Load: {forecast['day_1']['patient_load']}, Disease: {forecast['day_1']['dominant_disease']}")
        print(f"Day 2 - Load: {forecast['day_2']['patient_load']}, Disease: {forecast['day_2']['dominant_disease']}")
        print(f"Day 3 - Load: {forecast['day_3']['patient_load']}, Disease: {forecast['day_3']['dominant_disease']}")
    
    # Step 3: Load hospitals
    print("\n[Step 3] Loading hospitals in area...")
    hospitals_str = load_hospitals_by_pincode_impl(str(user_data["pincode"]))
    hospitals_data = json.loads(hospitals_str)
    print(f"Found {hospitals_data['total_hospitals_found']} hospitals, returning top {hospitals_data['hospitals_returned']}")
    
    # Prepare output for Agent 2
    output = {
        "patient_payload": user_data,
        "model_prediction": {
            "predicted_disease": prediction['predictions'][0]['predicted_disease_spike'],
            "risk_intensity": prediction['predictions'][0]['predicted_intensity'],
            "spike_probability": prediction['predictions'][0]['predicted_patient_load'],
            "dominant_risk": prediction['predictions'][0]['dominant_risk'],
            "disease_risks": prediction['predictions'][0]['disease_risks']
        },
        "hospitals_in_region": hospitals_data['hospitals']
    }
    
    print("\n[SUCCESS] Agent 1 completed successfully!")
    return output


# ===========================
# AGENT 2: Medical Recommendation Generator
# ===========================

def run_agent_2(agent1_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes Agent 2: Medical Recommendation Generator
    Uses LLM to analyze and generate recommendations
    """
    print("\n" + "="*60)
    print("[AGENT 2] Medical Recommendation Generator")
    print("="*60)
    
    # Create prompt for LLM
    system_prompt = """You are a medical recommendation system expert. 
    
Your task is to analyze patient health predictions and hospital capabilities to generate actionable recommendations.

You will receive:
1. Patient environmental data and health risk predictions
2. List of hospitals in the area with their capabilities

You must:
1. Map the predicted disease type to required medical specialties
2. Rank hospitals based on:
   - Specialty match
   - Bed capacity (total and ICU)
   - Staff availability (doctors, nurses)
   - Emergency readiness
   - Ventilator availability
   - Oxygen supply
   - Hospital rating
3. Generate specific recommendations for:
   - Patients (which hospital to go to, precautions)
   - Hospitals (prepare for surge, stock supplies)
   - Public health (area advisories)

Provide your response in JSON format with the following structure:
{
  "patient_recommendations": ["recommendation 1", "recommendation 2", ...],
  "hospital_recommendations": [
    {
      "hospital_name": "name",
      "hospital_id": id,
      "rank": 1,
      "match_score": 0.95,
      "reason": "why this hospital is recommended",
      "action": "what hospital should prepare"
    },
    ...
  ],
  "area_advisory": "public health message for the area",
  "severity_flag": "low/medium/high",
  "reasoning_summary": "brief explanation of your recommendations"
}
"""
    
    user_message = f"""Please analyze this data and generate recommendations:

PATIENT DATA:
{json.dumps(agent1_output['patient_payload'], indent=2)}

ML PREDICTION:
{json.dumps(agent1_output['model_prediction'], indent=2)}

AVAILABLE HOSPITALS (showing top 10):
{json.dumps(agent1_output['hospitals_in_region'][:10], indent=2)}

Generate comprehensive medical recommendations."""
    
    print("\n[AI] Analyzing with AI...")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    
    # Parse LLM response
    try:
        # Try to extract JSON from response
        response_text = response.content
        
        # Find JSON in response (between first { and last })
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            recommendations = json.loads(json_str)
        else:
            # Fallback if no JSON found
            recommendations = {
                "raw_response": response_text,
                "parsing_error": "Could not extract JSON from response"
            }
    except Exception as e:
        recommendations = {
            "raw_response": response.content,
            "parsing_error": str(e)
        }
    
    print(f"\n[RESULT] Generated {len(recommendations.get('hospital_recommendations', []))} hospital recommendations")
    print(f"Severity Assessment: {recommendations.get('severity_flag', 'unknown')}")
    
    print("\n[SUCCESS] Agent 2 completed successfully!")
    return recommendations


# ===========================
# AGENT 3: Feasibility & Practicality Validator
# ===========================

def run_agent_3(agent2_output: Dict[str, Any], agent1_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes Agent 3: Feasibility & Practicality Validator
    Validates and formats final output
    """
    print("\n" + "="*60)
    print("[AGENT 3] Feasibility & Practicality Validator")
    print("="*60)
    
    system_prompt = """You are a medical feasibility validator. 

Your task is to review medical recommendations and ensure they are:
1. Practical and actionable
2. Realistic given resource constraints
3. Safe and medically sound
4. Clear and user-friendly

You will receive recommendations from Agent 2. You must:
1. Validate each recommendation against hospital capabilities
2. Filter out unrealistic suggestions (e.g., recommending hospital with 0 ICU beds for critical cases)
3. Check for conflicts or impossibilities
4. Format output for UI consumption
5. Add warnings where needed

Provide your response in JSON format:
{
  "final_patient_guidance": ["clear action item 1", "action item 2", ...],
  "final_hospital_guidance": [
    {
      "hospital_name": "name",
      "recommendation": "what to do",
      "confidence": "high/medium/low",
      "validated": true/false,
      "reason": "why this is feasible/not feasible"
    }
  ],
  "top_hospitals_ui_format": [
    {
      "rank": 1,
      "name": "hospital name",
      "specialty": "relevant specialty",
      "available_beds": number,
      "icu_beds": number,
      "distance_category": "nearby/moderate/far",
      "recommendation": "brief reason to choose this hospital"
    }
  ],
  "warnings_or_caveats": ["warning 1", ...],
  "validation_summary": "overall assessment",
  "final_severity": "low/medium/high"
}
"""
    
    user_message = f"""Please validate these recommendations and create final UI-ready output:

AGENT 2 RECOMMENDATIONS:
{json.dumps(agent2_output, indent=2)}

ORIGINAL HOSPITAL DATA (for validation):
{json.dumps(agent1_output['hospitals_in_region'][:10], indent=2)}

PATIENT CONTEXT:
{json.dumps(agent1_output['model_prediction'], indent=2)}

Validate feasibility and format for UI."""
    
    print("\n[VALIDATION] Validating recommendations...")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]
    
    response = llm.invoke(messages)
    
    # Parse LLM response
    try:
        response_text = response.content
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            final_output = json.loads(json_str)
        else:
            final_output = {
                "raw_response": response_text,
                "parsing_error": "Could not extract JSON from response"
            }
    except Exception as e:
        final_output = {
            "raw_response": response.content,
            "parsing_error": str(e)
        }
    
    print(f"\n[RESULT] Validated {len(final_output.get('top_hospitals_ui_format', []))} hospitals for UI")
    print(f"Final Severity: {final_output.get('final_severity', 'unknown')}")
    print(f"Warnings: {len(final_output.get('warnings_or_caveats', []))}")
    
    print("\n[SUCCESS] Agent 3 completed successfully!")
    return final_output


# ===========================
# PIPELINE ORCHESTRATOR
# ===========================

def run_agent_pipeline():
    """
    Main orchestrator that runs all three agents in sequence
    """
    print("\n" + "="*60)
    print("STARTING THREE-AGENT LANGCHAIN PIPELINE")
    print("="*60)
    
    # Check backend status
    try:
        response = requests.get(f"{BACKEND_URL}/user_data", timeout=3)
        print(f"\n[INFO] Backend server is running at {BACKEND_URL}")
    except:
        print(f"\n[INFO] Backend server not available at {BACKEND_URL}")
        print("[INFO] Using mock data for demonstration")
    
    try:
        # Execute Agent 1
        agent1_result = run_agent_1()
        
        # Execute Agent 2
        agent2_result = run_agent_2(agent1_result)
        
        # Execute Agent 3
        agent3_result = run_agent_3(agent2_result, agent1_result)
        
        # Final output
        print("\n" + "="*60)
        print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\n[OUTPUT] FINAL OUTPUT SUMMARY:")
        print(json.dumps(agent3_result, indent=2))
        
        # Save to file
        with open("agent_pipeline_output.json", "w") as f:
            json.dump({
                "agent1_output": agent1_result,
                "agent2_output": agent2_result,
                "agent3_output": agent3_result
            }, f, indent=2, default=str)
        
        print("\n[SAVED] Full pipeline output saved to: agent_pipeline_output.json")
        
        return agent3_result
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the complete pipeline
    result = run_agent_pipeline()
    
    if result:
        print("\n[COMPLETE] All agents executed successfully!")
        print("\nTop 3 Recommended Hospitals:")
        for i, hospital in enumerate(result.get('top_hospitals_ui_format', [])[:3], 1):
            print(f"\n{i}. {hospital.get('name', 'Unknown')}")
            print(f"   Specialty: {hospital.get('specialty', 'N/A')}")
            print(f"   Available Beds: {hospital.get('available_beds', 'N/A')}")
            print(f"   Reason: {hospital.get('recommendation', 'N/A')}")
