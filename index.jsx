import React, { useState } from 'react';
import './styles.css'; // Import your existing styles

const App = () => {
  // State to track if the user is logged in or not
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // Function to handle login (for example purposes)
  const handleLogin = () => {
    setIsLoggedIn(true); // Simulate login
  };

  // Function to handle logout
  const handleLogout = () => {
    setIsLoggedIn(false); // Simulate logout
  };

  return (
    <div>
      <header>
        <div id="first">
          <img src="/logo2.avif" alt="Quiz Logo" className="logo" />
          <div className="search-and-login">
            <a href="#"><i className="fas fa-search search-icon"></i></a>

            {/* Conditionally render based on login status */}
            {isLoggedIn ? (
              <a href="/profile.html">
                <i className="fas fa-user-circle profile-circle-icon" title="Profile"></i>
              </a>
            ) : (
              <button onClick={handleLogin}>Log-in</button>
            )}

            <a href="https://quizbuddy.streamlit.app/" target="_blank" title="Chatbot">
              <i className="fas fa-comments chatbot-icon"></i>
            </a>
          </div>
        </div>
        <nav>
          <ul>
            <li><a href="#">Home</a></li>
            <li><a href="#">Quizzes</a></li>
            <li><a href="#">Leaderboard</a></li>
            <li><a href="#">Categories</a></li>
          </ul>
        </nav>
      </header>

      <main>
        <section className="quiz-section">
          <h1>Test your knowledge</h1>
          <p>How much do you know about sustainability? Test your knowledge with this short quiz.</p>
          <a href="/quiz.html">
            <button className="start-quiz-btn">Start Quiz</button>
          </a>
        </section>
      </main>

      <footer id="footer">
        <div className="footer-content">
          <a href="mailto:support@example.com" className="contact-us">Contact Us</a>
          <div className="social-icons">
            <a href="https://www.instagram.com" target="_blank">
              <i className="fab fa-instagram"></i>
            </a>
            <a href="mailto:support@example.com">
              <i className="fas fa-envelope"></i>
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
