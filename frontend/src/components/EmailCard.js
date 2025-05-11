import React from 'react';

const categoryColors = {
  Work: 'bg-blue-100 text-blue-800',
  Personal: 'bg-green-100 text-green-800',
  Finance: 'bg-yellow-100 text-yellow-800',
  Promotions: 'bg-purple-100 text-purple-800',
  Travel: 'bg-red-100 text-red-800',
  Spam: 'bg-gray-100 text-gray-800'
};

function EmailCard({ email, showCategory }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <h2 className="text-xl font-semibold text-gray-900">{email.subject}</h2>
        {showCategory && email.category && (
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium ${
              categoryColors[email.category] || categoryColors.Personal
            }`}
          >
            {email.category}
          </span>
        )}
      </div>
      <div className="text-sm text-gray-600 mb-4">
        <p><strong>From:</strong> {email.from}</p>
        <p><strong>Date:</strong> {email.date}</p>
      </div>
      <div className="flex justify-end">
        <a
          href={email.gmail_link}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:text-blue-800 text-sm font-medium"
        >
          View in Gmail →
        </a>
      </div>
    </div>
  );
}

export default EmailCard; 