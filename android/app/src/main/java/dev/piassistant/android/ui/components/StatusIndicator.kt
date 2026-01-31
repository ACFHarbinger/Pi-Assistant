package dev.piassistant.android.ui.components

import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import dev.piassistant.android.data.AgentState

@Composable
fun StatusIndicator(state: AgentState?, isConnected: Boolean) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(8.dp)
    ) {
        val color = if (!isConnected) {
            MaterialTheme.colorScheme.error
        } else when (state?.status) {
            "Running" -> Color(0xFF48BB78) // Green
            "Paused" -> Color(0xFFECC94B) // Yellow
            "Idle" -> Color(0xFF4299E1)   // Blue
            else -> MaterialTheme.colorScheme.outline
        }
        
        Surface(
            shape = MaterialTheme.shapes.small,
            color = color,
            modifier = Modifier.size(8.dp)
        ) {}
        
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = if (!isConnected) "Offline" else state?.status ?: "Connecting...",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}
