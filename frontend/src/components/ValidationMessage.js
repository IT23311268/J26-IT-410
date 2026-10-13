import React from 'react';

function ValidationMessage({ type, message }) {
  if (!message) return null;

  const icons = {
    error: '⚠️',
    success: '✅',
    info: 'ℹ️',
  };

  return (
    <div className={`validation-message ${type}`}>
      <span>{icons[type] || ''}</span>
      <span>{message}</span>
    </div>
  );
}

export default ValidationMessage;
