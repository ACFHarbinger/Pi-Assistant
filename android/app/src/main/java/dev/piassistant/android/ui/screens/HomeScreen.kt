package dev.piassistant.android.ui.screens

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.piassistant.android.ui.components.ChatBubble
import dev.piassistant.android.ui.components.PermissionCard
import dev.piassistant.android.ui.components.StatusIndicator
import dev.piassistant.android.viewmodel.ChatViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(viewModel: ChatViewModel = viewModel()) {
    val state by viewModel.agentState.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val isConnected by viewModel.isConnected.collectAsState()

    Column(modifier = Modifier.fillMaxSize()) {
        // Header
        CenterAlignedTopAppBar(
            title = { Text("Pi-Assistant") },
            actions = {
                StatusIndicator(state = state, isConnected = isConnected)
            }
        )

        // Chat List
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .padding(16.dp),
            contentPadding = PaddingValues(bottom = 16.dp)
        ) {
            items(messages) { msg ->
                ChatBubble(msg)
            }
        }

        // Permission Request Overlay
        state?.data?.awaiting_permission?.let { req ->
            PermissionCard(
                request = req,
                onApprove = { remember -> viewModel.approvePermission(req.id, remember) },
                onDeny = { viewModel.denyPermission(req.id) }
            )
        }

        // Input Area
        InputArea(
            onSend = { text ->
                if (state?.status == "Idle") {
                    viewModel.startTask(text)
                } else if (state?.data?.question != null) {
                    viewModel.sendAnswer(text)
                }
            },
            enabled = isConnected && (state?.status == "Idle" || state?.data?.question != null)
        )
    }
}

@Composable
fun InputArea(onSend: (String) -> Unit, enabled: Boolean) {
    var text by remember { mutableStateOf("") }

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        OutlinedTextField(
            value = text,
            onValueChange = { text = it },
            modifier = Modifier.weight(1f),
            enabled = enabled,
            placeholder = { Text("Enter task or reply...") }
        )
        Spacer(modifier = Modifier.width(8.dp))
        Button(
            onClick = {
                if (text.isNotBlank()) {
                    onSend(text)
                    text = ""
                }
            },
            enabled = enabled && text.isNotBlank()
        ) {
            Text("Send")
        }
    }
}
