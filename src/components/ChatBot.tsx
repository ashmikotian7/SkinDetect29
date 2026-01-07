import React, { useState, useRef, useEffect } from "react";

interface Message {
  sender: "user" | "bot";
  text: string;
}

interface QAItem {
  response: string;
  keywords: string[];
}

const SMART_QA_DB: QAItem[] = [
  { response: "Skin cancer is a disease where skin cells grow uncontrollably, often due to UV exposure.", keywords: ["what is skin cancer", "explain skin cancer"] },
  { response: "Skin cancer is common, with millions of cases worldwide every year.", keywords: ["how common is skin cancer", "skin cancer prevalence"] },
  { response: "Anyone can get skin cancer, but fair-skinned people and those with sun exposure are at higher risk.", keywords: ["who is at risk", "risk factors"] },
  { response: "Yes, people with dark skin can get skin cancer, often in areas not exposed to the sun.", keywords: ["dark skin risk", "skin cancer in dark skin"] },
  { response: "Some types of skin cancer can be hereditary due to genetic mutations.", keywords: ["hereditary", "family history"] },
  { response: "Children can get skin cancer, though it is rare; sun protection is important from birth.", keywords: ["children risk", "skin cancer kids"] },
  { response: "Melanoma is the most aggressive type, while basal and squamous cell cancers are generally less aggressive.", keywords: ["types severity", "dangerous skin cancer"] },
  { response: "Yes, skin cancer can be cured, especially if detected early.", keywords: ["can be cured", "treatment success"] },
  { response: "Skin cancer can recur; ongoing monitoring is essential.", keywords: ["recurrence", "come back"] },
  { response: "Melanoma is serious because it spreads quickly if untreated.", keywords: ["melanoma seriousness", "melanoma danger"] },
  { response: "Early signs include new growths, sores that don’t heal, unusual moles, or changes in existing moles.", keywords: ["early signs", "first signs", "warning signs"] },
  { response: "ABCDE rule: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving changes.", keywords: ["ABCDE rule", "melanoma warning","rules"] },
  { response: "Persistent sores, bleeding moles, itching, or rapidly growing lesions are common symptoms.", keywords: ["symptoms", "signs", "itching", "bleeding mole"] },
  { response: "Skin cancer can appear under nails, on scalp, face, arms, or any sun-exposed areas.", keywords: ["hidden areas", "nails", "scalp", "face"] },
  { response: "Fair skin increases risk because of lower melanin protection.", keywords: ["fair skin risk"] },
  { response: "UV exposure, tanning beds, and previous sunburns increase risk.", keywords: ["UV exposure", "tanning bed risk", "sunburn"] },
  { response: "Family history of skin cancer increases susceptibility.", keywords: ["family history risk"] },
  { response: "Age increases risk due to cumulative sun exposure.", keywords: ["age risk"] },
  { response: "Weakened immune systems increase vulnerability to skin cancer.", keywords: ["immune system risk"] },
  { response: "Certain chemicals and medications may increase risk.", keywords: ["chemical risk", "medication risk"] },
  { response: "Sunscreen SPF 30+ should be applied 15-30 mins before sun exposure.", keywords: ["sunscreen SPF", "apply sunscreen"] },
  { response: "Reapply sunscreen every 2 hours or after swimming/sweating.", keywords: ["reapply sunscreen"] },
  { response: "Wear protective clothing, hats, and sunglasses to reduce UV exposure.", keywords: ["protective clothing", "hats", "sunglasses"] },
  { response: "Avoid tanning beds; they significantly increase melanoma risk.", keywords: ["avoid tanning beds", "artificial UV"] },
  { response: "Seek shade during peak sun hours, 10am-4pm.", keywords: ["shade", "peak sun hours"] },
  { response: "Moisturizing and gentle skin care help maintain healthy skin.", keywords: ["skin care tips", "moisturize","tips","care"] },
  { response: "Hydrate, eat a balanced diet, and avoid smoking to support skin health.", keywords: ["diet", "hydration", "healthy skin"] },
  { response: "Check all areas of your skin monthly using mirrors for hard-to-see spots.", keywords: ["self-exam", "monthly check"] },
  { response: "Report any changing, bleeding, or non-healing lesions to a dermatologist immediately.", keywords: ["when to see doctor", "urgent signs"] },
  { response: "Basal cell carcinoma grows slowly and rarely spreads.", keywords: ["basal cell carcinoma", "BCC"] },
  { response: "Squamous cell carcinoma can spread if untreated.", keywords: ["squamous cell carcinoma", "SCC"] },
  { response: "Melanoma is aggressive and has a high risk of metastasis.", keywords: ["melanoma"] },
  { response: "Merkel cell carcinoma is rare and aggressive.", keywords: ["merkel cell carcinoma"] },
  { response: "Actinic keratosis is a precancerous scaly patch from sun exposure.", keywords: ["actinic keratosis", "pre-cancer"] },
  { response: "Treatment options include surgery, cryotherapy, topical therapy, radiation, immunotherapy, and targeted therapy.", keywords: ["treatment options", "therapy", "surgery"] },
  { response: "Mohs surgery removes skin cancer layer by layer while sparing healthy tissue.", keywords: ["Mohs surgery"] },
  { response: "Cryotherapy freezes abnormal cells with liquid nitrogen.", keywords: ["cryotherapy"] },
  { response: "Topical therapies like imiquimod stimulate immune response in affected areas.", keywords: ["topical therapy"] },
  { response: "Radiation therapy uses high-energy rays to kill cancer cells.", keywords: ["radiation therapy"] },
  { response: "Immunotherapy boosts the immune system to attack cancer cells.", keywords: ["immunotherapy"] },
  { response: "Targeted therapy uses drugs aimed at specific cancer mutations.", keywords: ["targeted therapy"] },
  { response: "Photodynamic therapy uses light-activated medication to destroy cancer cells.", keywords: ["photodynamic therapy"] },
  { response: "Skin biopsy is the definitive way to diagnose skin cancer.", keywords: ["biopsy", "diagnosis test"] },
  { response: "Dermatoscopic exams help distinguish benign from malignant lesions.", keywords: ["dermatoscopy", "dermatoscope"] },
  { response: "Yes, multiple skin cancers can appear at different times or sites.", keywords: ["multiple cancers"] },
  { response: "UV index forecasts help plan safe outdoor activities.", keywords: ["UV index"] },
  { response: "Even on cloudy days, UV rays can cause skin damage.", keywords: ["cloudy days UV"] },
  { response: "Skin cancer can appear anywhere, including scalp, soles, lips, or ears.", keywords: ["unusual areas", "scalp", "soles", "ears"] },
  { response: "Vitamin D can be safely obtained through diet or supplements if sun exposure is limited.", keywords: ["vitamin D"] },
  { response: "Support groups and counseling help patients cope with diagnosis.", keywords: ["emotional support", "support groups"] },
  { response: "Document moles and skin changes using photos to track progress.", keywords: ["monitor moles", "photograph moles"] },
  { response: "Avoid sun exposure immediately after certain treatments like surgery or photodynamic therapy.", keywords: ["sun after treatment", "post treatment care"] },
  { response: "Annual skin checks are recommended for high-risk adults or those over 40.", keywords: ["annual checkup", "high-risk monitoring"] },
  { response: "Melanoma thickness (Breslow depth) affects prognosis and survival.", keywords: ["melanoma thickness", "Breslow depth"] },
  { response: "Even small lesions can be dangerous; early evaluation is critical.", keywords: ["small lesion risk", "early detection"] },
  { response: "Yes, moles may change over time; track for shape, color, and size changes.", keywords: ["mole changes", "monitor moles"] },
  { response: "Skin cancer rarely affects internal organs directly but can metastasize.", keywords: ["spread internal organs"] },
  { response: "Teledermatology allows remote assessment of skin lesions for initial guidance.", keywords: ["teledermatology", "online consultation"] },
  { response: "Scar management includes moisturizing, sun protection, and silicone sheets if recommended.", keywords: ["scar management", "post-surgery care"] },
  { response: "Clothing with UPF rating and tightly woven fabrics reduces UV penetration.", keywords: ["UPF clothing", "protective fabrics"] },
  { response: "Sunglasses with UVA/UVB protection protect eyes and surrounding skin.", keywords: ["eye protection", "sunglasses"] },
  { response: "Avoid tanning oils; they increase UV absorption.", keywords: ["tanning oils", "sun risk"] },
  { response: "Even minimal sun exposure can accumulate over time; use protection consistently.", keywords: ["cumulative UV", "sun exposure"] },
  { response: "Avoid direct sun for infants; hats and protective clothing are essential.", keywords: ["infants sun protection"] },
  { response: "Genetic counseling may help families with strong history of melanoma.", keywords: ["genetic counseling", "family risk"] },
  { response: "Non-melanoma skin cancers like BCC and SCC have high cure rates when detected early.", keywords: ["non-melanoma prognosis"] },
  { response: "Yes, repeated sunburns in childhood significantly increase adult risk.", keywords: ["childhood sunburn", "adult risk"] },
  { response: "Skin cancer awareness months help promote education and early detection.", keywords: ["awareness month", "education"] },
  { response: "Avoid sun exposure during peak hours and use shade structures when outdoors.", keywords: ["shade", "peak sun hours"] },
  { response: "Skin cancer can affect mental health; counseling is recommended if anxious or depressed.", keywords: ["mental health", "psychological impact"] },
  { response: "Safe sun exposure involves protective clothing, sunscreen, hats, and avoiding peak UV times.", keywords: ["safe sun exposure"] },
];

const GREETINGS = ["hi", "hello", "hey", "hii", "hiii", "hyyyy", "hola", "heyy"];

const Chatbot: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: "bot",
      text: "Hi! I’m your AI assistant for skin care and skin cancer awareness. How can I help you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [typingQueue, setTypingQueue] = useState<string[]>([]);
  const [darkMode, setDarkMode] = useState(false);
  const chatBodyRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [messages, typingQueue, isTyping]);

  // Process typing queue
  useEffect(() => {
    if (typingQueue.length === 0) {
      setIsTyping(false);
      return;
    }

    setIsTyping(true);
    const timer = setTimeout(() => {
      const nextLine = typingQueue[0];
      setMessages((prev) => [...prev, { sender: "bot", text: nextLine }]);
      setTypingQueue((prev) => prev.slice(1));
    }, 500);

    return () => clearTimeout(timer);
  }, [typingQueue]);

  // Smart answer finder
  const findSmartAnswer = (message: string): string | null => {
    const lowerMsg = message.toLowerCase();
    for (const item of SMART_QA_DB) {
      if (item.keywords.some((kw) => lowerMsg.includes(kw))) {
        return item.response;
      }
    }
    return null;
  };

  const sendMessage = (msg?: string) => {
    const message = (msg || input).trim();
    if (!message) return;
    setMessages((prev) => [...prev, { sender: "user", text: message }]);
    setInput("");

    const lowerMessage = message.toLowerCase();

    // Greeting detection
    if (GREETINGS.some((g) => lowerMessage.includes(g))) {
      setTypingQueue([
        "Hello! 👋 I’m your AI assistant for skin care and skin cancer awareness.",
        "You can ask me about symptoms, prevention, skin care tips, and early warning signs.",
        "Feel free to ask anything related to skin or skin cancer.",
      ]);
      return;
    }

    // Smart QA matching
    const answer = findSmartAnswer(message);
    if (answer) {
      const lines = answer.split("\n").filter(Boolean);
      setTypingQueue(lines);
    } else {
      setTypingQueue([
        "I specialize in skin care and skin cancer topics.",
        "Please ask about skin care, prevention, symptoms, or types of skin cancer.",
      ]);
    }
  };

  const sendFAQ = (question: string) => sendMessage(question);

  return (
    <>
      {/* Floating Chat Button */}
      <div
        style={{
          position: "fixed",
          bottom: 20,
          right: 20,
          width: 60,
          height: 60,
          borderRadius: "50%",
          backgroundColor: "#25D366",
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          cursor: "pointer",
          fontSize: 28,
          color: "white",
          boxShadow: "0 4px 8px rgba(0,0,0,0.2)",
          zIndex: 1000,
        }}
        onClick={() => setIsOpen((prev) => !prev)}
      >
        💬
      </div>

      {/* Chatbox */}
      {isOpen && (
        <div
          style={{
            position: "fixed",
            bottom: 90,
            right: 20,
            width: 360,
            maxHeight: 550,
            backgroundColor: darkMode ? "#2C2F33" : "white",
            borderRadius: 10,
            boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
            fontFamily: "Arial, sans-serif",
            color: darkMode ? "white" : "black",
            zIndex: 1000,
          }}
        >
          {/* Header */}
          <div
            style={{
              backgroundColor: "#128C7E",
              color: "white",
              padding: "10px",
              fontWeight: "bold",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span>Skin Cancer Assistant</span>
            <div>
              <span
                style={{ cursor: "pointer", marginRight: 10 }}
                onClick={() => setDarkMode((prev) => !prev)}
                title="Toggle Dark/Light Mode"
              >
                {darkMode ? "🌙" : "☀️"}
              </span>
              <span style={{ cursor: "pointer" }} onClick={() => setIsOpen(false)}>
                ✖
              </span>
            </div>
          </div>

          {/* Chat Body */}
          <div
            style={{
              padding: 10,
              overflowY: "auto",
              flex: 1,
              backgroundColor: darkMode ? "#23272A" : "#ECE5DD",
            }}
            ref={chatBodyRef}
          >
            {messages.map((m, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  justifyContent: m.sender === "user" ? "flex-end" : "flex-start",
                  margin: "5px 0",
                }}
              >
                {m.sender === "bot" && (
                  <img
                    src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png"
                    alt="bot"
                    style={{ width: 32, height: 32, borderRadius: "50%", marginRight: 8 }}
                  />
                )}
                <div
                  style={{
                    backgroundColor: m.sender === "user" ? "#DCF8C6" : darkMode ? "#2C2F33" : "white",
                    padding: "8px 12px",
                    borderRadius: 15,
                    maxWidth: "80%",
                    lineHeight: 1.5,
                    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {m.text}
                </div>
              </div>
            ))}

            {isTyping && (
              <div style={{ display: "flex", alignItems: "center", margin: "5px 0" }}>
                <img
                  src="https://cdn-icons-png.flaticon.com/512/4712/4712027.png"
                  alt="bot"
                  style={{ width: 32, height: 32, borderRadius: "50%", marginRight: 8 }}
                />
                <div className="typing-dots" style={{ display: "flex" }}>
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      backgroundColor: "#999",
                      borderRadius: "50%",
                      margin: "0 2px",
                      animation: "blink 1s infinite 0.2s",
                    }}
                  ></span>
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      backgroundColor: "#999",
                      borderRadius: "50%",
                      margin: "0 2px",
                      animation: "blink 1s infinite 0.4s",
                    }}
                  ></span>
                </div>
              </div>
            )}
          </div>

          {/* Input */}
          <div style={{ display: "flex", borderTop: "1px solid #ccc" }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                if (e.key === "Enter") sendMessage();
              }}
              placeholder="Type your message..."
              style={{
                flex: 1,
                border: "none",
                padding: 10,
                fontSize: 14,
                outline: "none",
                color: darkMode ? "white" : "black",
                backgroundColor: darkMode ? "#2C2F33" : "white",
              }}
            />
            <button
              onClick={() => sendMessage()}
              style={{
                backgroundColor: "#25D366",
                border: "none",
                color: "white",
                padding: "0 15px",
                cursor: "pointer",
                fontSize: 16,
              }}
            >
              Send
            </button>
          </div>

          {/* FAQ Buttons */}
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "8px",
              padding: "10px",
              borderTop: "1px solid #ccc",
              backgroundColor: darkMode ? "#2C2F33" : "#f0f0f0",
              justifyContent: "center",
            }}
          >
            {["Early signs", "Prevention", "Skin care tips", "Symptoms", "Treatment options"].map(
              (q) => (
                <button
                  key={q}
                  onClick={() => sendFAQ(q)}
                  style={{
                    padding: "6px 12px",
                    fontSize: 13,
                    cursor: "pointer",
                    borderRadius: 20,
                    border: "none",
                    background: darkMode ? "#128C7E" : "#25D366",
                    color: "white",
                    fontWeight: "bold",
                    boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
                    transition: "all 0.2s",
                  }}
                  onMouseOver={(e) =>
                    (e.currentTarget.style.backgroundColor = darkMode ? "#0F7466" : "#1EB854")
                  }
                  onMouseOut={(e) =>
                    (e.currentTarget.style.backgroundColor = darkMode ? "#128C7E" : "#25D366")
                  }
                >
                  {q}
                </button>
              )
            )}
          </div>
        </div>
      )}

      <style>
        {`
          @keyframes blink {
            0%, 80%, 100% { opacity:0; }
            40% { opacity:1; }
          }
        `}
      </style>
    </>
  );
};

export default Chatbot;
