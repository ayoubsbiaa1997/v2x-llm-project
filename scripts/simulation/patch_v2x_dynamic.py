#!/usr/bin/env python3
"""
Création contrôlée de v2x-5g-dynamic.cc à partir de v2x-5g-sumo.cc.

Le fichier source original n'est jamais modifié.
Chaque remplacement doit apparaître exactement une fois, sinon le script
s'arrête afin d'éviter de produire un programme C++ incohérent.
"""

from __future__ import annotations

from pathlib import Path
import re


SOURCE = Path.home() / "ns-3-dev/scratch/v2x-5g-sumo.cc"
OUTPUT = Path.home() / "ns-3-dev/scratch/v2x-5g-dynamic.cc"


def replace_once(
    text: str,
    old: str,
    new: str,
    description: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{description}: motif trouvé {count} fois au lieu d'une."
        )

    return text.replace(old, new, 1)


def replace_regex_once(
    text: str,
    pattern: str,
    replacement: str,
    description: str,
) -> str:
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )

    if count != 1:
        raise RuntimeError(
            f"{description}: bloc trouvé {count} fois au lieu d'une."
        )

    return updated


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {SOURCE}"
        )

    text = SOURCE.read_text(encoding="utf-8")

    # ---------------------------------------------------------
    # 1. Includes explicites
    # ---------------------------------------------------------
    text = replace_once(
        text,
        '#include <fstream>\n#include <iomanip>',
        '#include <algorithm>\n'
        '#include <fstream>\n'
        '#include <iomanip>\n'
        '#include <map>\n'
        '#include <string>',
        "Ajout des includes",
    )

    # ---------------------------------------------------------
    # 2. Logger QoS par fenêtres et changement dynamique du débit
    # ---------------------------------------------------------
    global_block = r'''
std::ofstream qosTemporalFile;
std::map<FlowId, FlowMonitor::FlowStats> previousFlowStats;

std::string currentTrafficPhase = "STATIC";
uint32_t currentLambdaBe = 0;

NS_LOG_COMPONENT_DEFINE("CttcNrDemo");


void
UpdateUdpClientRate(
    ApplicationContainer applications,
    uint32_t lambda,
    std::string phase)
{
    if (lambda == 0)
    {
        NS_FATAL_ERROR("Le débit lambda doit être strictement positif.");
    }

    const Time interval =
        Seconds(1.0 / static_cast<double>(lambda));

    for (uint32_t i = 0; i < applications.GetN(); ++i)
    {
        Ptr<UdpClient> client =
            DynamicCast<UdpClient>(applications.Get(i));

        if (client == nullptr)
        {
            NS_FATAL_ERROR(
                "Une application dynamique n'est pas un UdpClient."
            );
        }

        client->SetAttribute(
            "Interval",
            TimeValue(interval)
        );
    }

    currentTrafficPhase = phase;
    currentLambdaBe = lambda;

    std::cout
        << "TRAFFIC_PHASE"
        << " time=" << Simulator::Now().GetSeconds()
        << " phase=" << currentTrafficPhase
        << " lambdaBe=" << currentLambdaBe
        << std::endl;
}


void
LogTemporalQos(
    Ptr<FlowMonitor> monitor,
    double interval)
{
    monitor->CheckForLostPackets();

    const auto stats = monitor->GetFlowStats();

    double totalThroughputMbps = 0.0;
    double totalDelayWindowMs = 0.0;
    double totalJitterWindowMs = 0.0;

    uint64_t totalDeltaTxPackets = 0;
    uint64_t totalDeltaRxPackets = 0;
    uint64_t totalDeltaJitterSamples = 0;

    for (const auto& flow : stats)
    {
        const FlowId flowId = flow.first;
        const FlowMonitor::FlowStats& current = flow.second;

        FlowMonitor::FlowStats previous{};

        const auto previousIt =
            previousFlowStats.find(flowId);

        if (previousIt != previousFlowStats.end())
        {
            previous = previousIt->second;
        }

        const uint64_t deltaTxPackets =
            current.txPackets >= previous.txPackets
                ? current.txPackets - previous.txPackets
                : current.txPackets;

        const uint64_t deltaRxPackets =
            current.rxPackets >= previous.rxPackets
                ? current.rxPackets - previous.rxPackets
                : current.rxPackets;

        const uint64_t deltaRxBytes =
            current.rxBytes >= previous.rxBytes
                ? current.rxBytes - previous.rxBytes
                : current.rxBytes;

        const uint64_t currentJitterSamples =
            current.rxPackets > 0
                ? current.rxPackets - 1
                : 0;

        const uint64_t previousJitterSamples =
            previous.rxPackets > 0
                ? previous.rxPackets - 1
                : 0;

        const uint64_t deltaJitterSamples =
            currentJitterSamples >= previousJitterSamples
                ? currentJitterSamples - previousJitterSamples
                : currentJitterSamples;

        const Time deltaDelay =
            current.delaySum - previous.delaySum;

        const Time deltaJitter =
            current.jitterSum - previous.jitterSum;

        totalThroughputMbps +=
            static_cast<double>(deltaRxBytes) *
            8.0 /
            (interval * 1000000.0);

        totalDelayWindowMs +=
            deltaDelay.GetMilliSeconds();

        totalJitterWindowMs +=
            deltaJitter.GetMilliSeconds();

        totalDeltaTxPackets += deltaTxPackets;
        totalDeltaRxPackets += deltaRxPackets;
        totalDeltaJitterSamples += deltaJitterSamples;

        previousFlowStats[flowId] = current;
    }

    double meanDelayMs = 0.0;
    double meanJitterMs = 0.0;
    double packetLossPercent = 0.0;
    double pdrPercent = 0.0;

    if (totalDeltaRxPackets > 0)
    {
        meanDelayMs =
            totalDelayWindowMs /
            static_cast<double>(totalDeltaRxPackets);
    }

    if (totalDeltaJitterSamples > 0)
    {
        meanJitterMs =
            totalJitterWindowMs /
            static_cast<double>(totalDeltaJitterSamples);
    }

    if (totalDeltaTxPackets > 0)
    {
        pdrPercent =
            100.0 *
            static_cast<double>(totalDeltaRxPackets) /
            static_cast<double>(totalDeltaTxPackets);

        pdrPercent = std::max(
            0.0,
            std::min(100.0, pdrPercent)
        );

        packetLossPercent =
            100.0 - pdrPercent;
    }

    const double congestionIndex = std::max(
        0.0,
        std::min(
            1.0,
            0.35 * (meanDelayMs / 30.0) +
            0.25 * (packetLossPercent / 20.0) +
            0.20 * (meanJitterMs / 5.0)
        )
    );

    if (qosTemporalFile.is_open())
    {
        qosTemporalFile
            << Simulator::Now().GetSeconds() << ","
            << currentTrafficPhase << ","
            << currentLambdaBe << ","
            << totalThroughputMbps << ","
            << meanDelayMs << ","
            << meanJitterMs << ","
            << packetLossPercent << ","
            << pdrPercent << ","
            << congestionIndex
            << std::endl;
    }

    Simulator::Schedule(
        Seconds(interval),
        &LogTemporalQos,
        monitor,
        interval
    );
}


int
main(int argc, char* argv[])
'''

    text = replace_regex_once(
        text,
        r'std::ofstream qosTemporalFile;\s*'
        r'NS_LOG_COMPONENT_DEFINE\("CttcNrDemo"\);'
        r'.*?'
        r'\nint\s*\nmain\(int argc, char\* argv\[\]\)',
        global_block.strip(),
        "Remplacement du logger temporel",
    )

    # ---------------------------------------------------------
    # 3. Paramètres dynamiques
    # ---------------------------------------------------------
    text = replace_once(
        text,
        '    uint32_t lambdaBe = 10000;',
        '''    uint32_t lambdaBe = 10000;

    // Paramètres de la campagne dynamique.
    bool dynamicTraffic = false;

    uint32_t stableLambdaBe = 100;
    uint32_t mediumLambdaBe = 1000;
    uint32_t highLambdaBe = 5000;
    uint32_t recoveryLambdaBe = 200;

    double phase1End = 12.0;
    double phase2End = 24.0;
    double phase3End = 36.0;

    double qosLogInterval = 0.5;

    uint32_t dynamicSeed = 12345;
    uint32_t dynamicRun = 1;''',
        "Ajout des variables dynamiques",
    )

    command_line_parameters = '''    cmd.AddValue(
        "dynamicTraffic",
        "Activer les changements dynamiques de trafic",
        dynamicTraffic);

    cmd.AddValue(
        "stableLambdaBe",
        "Debit BE pendant la phase stable",
        stableLambdaBe);

    cmd.AddValue(
        "mediumLambdaBe",
        "Debit BE pendant la phase intermediaire",
        mediumLambdaBe);

    cmd.AddValue(
        "highLambdaBe",
        "Debit BE pendant la phase de congestion",
        highLambdaBe);

    cmd.AddValue(
        "recoveryLambdaBe",
        "Debit BE pendant la phase de recuperation",
        recoveryLambdaBe);

    cmd.AddValue(
        "phase1End",
        "Fin de la phase stable en secondes",
        phase1End);

    cmd.AddValue(
        "phase2End",
        "Fin de la phase intermediaire en secondes",
        phase2End);

    cmd.AddValue(
        "phase3End",
        "Fin de la phase de congestion en secondes",
        phase3End);

    cmd.AddValue(
        "qosLogInterval",
        "Intervalle de mesure QoS en secondes",
        qosLogInterval);

    cmd.AddValue(
        "dynamicSeed",
        "Seed aleatoire ns-3",
        dynamicSeed);

    cmd.AddValue(
        "dynamicRun",
        "Numero de run aleatoire ns-3",
        dynamicRun);

'''

    text = replace_once(
        text,
        '    cmd.AddValue("simTime",',
        command_line_parameters
        + '    cmd.AddValue("simTime",',
        "Ajout des paramètres CommandLine",
    )

    validation_block = '''    cmd.Parse(argc, argv);

    if (lambdaULL == 0 || lambdaBe == 0)
    {
        NS_FATAL_ERROR(
            "lambdaULL et lambdaBe doivent être strictement positifs."
        );
    }

    if (qosLogInterval <= 0.0)
    {
        NS_FATAL_ERROR(
            "qosLogInterval doit être strictement positif."
        );
    }

    if (dynamicTraffic)
    {
        if (
            stableLambdaBe == 0 ||
            mediumLambdaBe == 0 ||
            highLambdaBe == 0 ||
            recoveryLambdaBe == 0
        )
        {
            NS_FATAL_ERROR(
                "Tous les débits dynamiques doivent être positifs."
            );
        }

        if (
            udpAppStartTime.GetSeconds() >= phase1End ||
            phase1End >= phase2End ||
            phase2End >= phase3End ||
            phase3End >= simTime.GetSeconds()
        )
        {
            NS_FATAL_ERROR(
                "Ordre temporel invalide : "
                "udpAppStart < phase1End < phase2End < "
                "phase3End < simTime."
            );
        }
    }

    RngSeedManager::SetSeed(dynamicSeed);
    RngSeedManager::SetRun(dynamicRun);

    currentTrafficPhase =
        dynamicTraffic ? "STABLE" : "STATIC";

    currentLambdaBe =
        dynamicTraffic ? stableLambdaBe : lambdaBe;'''

    text = replace_once(
        text,
        '    cmd.Parse(argc, argv);',
        validation_block,
        "Validation des paramètres dynamiques",
    )

    # ---------------------------------------------------------
    # 4. Débit initial du trafic BE
    # ---------------------------------------------------------
    text = replace_once(
        text,
        '    UdpClientHelper dlClientLowLat;',
        '''    const uint32_t initialLambdaBe =
        dynamicTraffic ? stableLambdaBe : lambdaBe;

    UdpClientHelper dlClientLowLat;''',
        "Définition du débit BE initial",
    )

    text = replace_once(
        text,
        'Seconds(1.0 / lambdaBe)',
        'Seconds(1.0 / static_cast<double>(initialLambdaBe))',
        "Utilisation du débit BE initial",
    )

    # ---------------------------------------------------------
    # 5. Séparation des conteneurs client
    # ---------------------------------------------------------
    text = replace_once(
        text,
        '    ApplicationContainer clientApps;',
        '''    ApplicationContainer clientApps;
    ApplicationContainer lowLatClientApps;
    ApplicationContainer voiceClientApps;''',
        "Création des conteneurs clients séparés",
    )

    text = replace_once(
        text,
        '        clientApps.Add(dlClientLowLat.Install(remoteHost));',
        '''        ApplicationContainer installedLowLat =
            dlClientLowLat.Install(remoteHost);

        lowLatClientApps.Add(installedLowLat);
        clientApps.Add(installedLowLat);''',
        "Installation des clients ULL",
    )

    text = replace_once(
        text,
        '        clientApps.Add(dlClientVoice.Install(remoteHost));',
        '''        ApplicationContainer installedVoice =
            dlClientVoice.Install(remoteHost);

        voiceClientApps.Add(installedVoice);
        clientApps.Add(installedVoice);''',
        "Installation des clients BE",
    )

    # ---------------------------------------------------------
    # 6. Planification des changements de régime
    # ---------------------------------------------------------
    dynamic_schedule = '''    clientApps.Stop(simTime);

    if (dynamicTraffic)
    {
        Simulator::Schedule(
            Seconds(phase1End),
            &UpdateUdpClientRate,
            voiceClientApps,
            mediumLambdaBe,
            std::string("MEDIUM")
        );

        Simulator::Schedule(
            Seconds(phase2End),
            &UpdateUdpClientRate,
            voiceClientApps,
            highLambdaBe,
            std::string("HIGH")
        );

        Simulator::Schedule(
            Seconds(phase3End),
            &UpdateUdpClientRate,
            voiceClientApps,
            recoveryLambdaBe,
            std::string("RECOVERY")
        );
    }'''

    text = replace_once(
        text,
        '    clientApps.Stop(simTime);',
        dynamic_schedule,
        "Planification des changements de trafic",
    )

    # ---------------------------------------------------------
    # 7. CSV temporel propre à chaque run
    # ---------------------------------------------------------
    old_qos_block = '''qosTemporalFile.open("qos_temporal_windows.csv",
                     std::ofstream::out | std::ofstream::trunc);

qosTemporalFile
    << "time_s,"
    << "throughput_mbps,"
    << "delay_ms,"
    << "jitter_ms,"
    << "packet_loss,"
    << "pdr,"
    << "congestion_index"
    << std::endl;

Simulator::Schedule(
    MilliSeconds(500),
    &LogTemporalQos,
    monitor,
    0.5
);'''

    new_qos_block = '''const std::string qosTemporalPath =
    outputDir + "/" + simTag + "_qos_temporal.csv";

qosTemporalFile.open(
    qosTemporalPath,
    std::ofstream::out | std::ofstream::trunc
);

if (!qosTemporalFile.is_open())
{
    NS_FATAL_ERROR(
        "Impossible d'ouvrir le fichier QoS : "
        << qosTemporalPath
    );
}

qosTemporalFile
    << std::fixed
    << std::setprecision(6)
    << "time_s,"
    << "phase,"
    << "lambda_be,"
    << "throughput_mbps,"
    << "delay_ms,"
    << "jitter_ms,"
    << "packet_loss,"
    << "pdr,"
    << "congestion_index"
    << std::endl;

std::cout
    << "QOS_TEMPORAL_FILE=" << qosTemporalPath
    << std::endl;

Simulator::Schedule(
    udpAppStartTime + Seconds(qosLogInterval),
    &LogTemporalQos,
    monitor,
    qosLogInterval
);'''

    text = replace_once(
        text,
        old_qos_block,
        new_qos_block,
        "Création du CSV temporel par run",
    )

    required_markers = [
        "UpdateUdpClientRate",
        "previousFlowStats",
        "dynamicTraffic",
        "stableLambdaBe",
        "qosLogInterval",
        "_qos_temporal.csv",
    ]

    missing_markers = [
        marker
        for marker in required_markers
        if marker not in text
    ]

    if missing_markers:
        raise RuntimeError(
            f"Marqueurs absents après patch : {missing_markers}"
        )

    OUTPUT.write_text(text, encoding="utf-8")

    print("Fichier source préservé :", SOURCE)
    print("Nouveau fichier créé    :", OUTPUT)
    print("Taille                   :", OUTPUT.stat().st_size, "octets")


if __name__ == "__main__":
    main()
