using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Random = UnityEngine.Random;

public enum GestureType
{
    Fireball,
    Frostball,
    Shoot,
    Nothing
}

public class Gesture
{
    public float Percentage { get; set; }
    public GestureType Type { get; set; }

    public Gesture()
    {
        
    }

    public Gesture(string name, float percentage)
    {
        Percentage = Mathf.Clamp(percentage, 0, 100);
        Type = (GestureType)int.Parse(name);
    }
}

public class MainGameManager : MonoBehaviour
{
    [SerializeField]
    private string targetNameInHieracy;

    [SerializeField]
    int concurrentParticles = 20;

    [SerializeField]
    GameObject channelFireballParticlePrefab, fireballPrefab;

    private GameObject targetHand;
    private GestureType activeSpell;
    private List<GameObject> activeParticles;

    private void Awake()
    {
        activeParticles = new List<GameObject>();
        HandThroughLeap.HandGesturePercentageEvent += OnEvent;
        activeSpell = GestureType.Nothing;
    }

    private void Start()
    {
        targetHand = GameObject.Find(targetNameInHieracy);
    }

    private void Update()
    {
        if (targetHand == null)
        {
            targetHand = GameObject.Find(targetNameInHieracy);
        }
    }

    private void OnEvent(Gesture g)
    {
        switch (g.Type)
        {
            case GestureType.Fireball:
                {
                    activeSpell = GestureType.Fireball;
                    ChannelFireball();
                    break;
                }
            case GestureType.Frostball:
                {
                    break;
                }
            case GestureType.Shoot:
                {
                    if(activeSpell == GestureType.Fireball)
                    {
                        ShootFireball();
                    }
                    break;
                }
            default:
                break;
        }
    }

    private void ChannelFireball()
    {
        for (int i = 0; i < concurrentParticles; i++)
        {
            var particle = Instantiate(channelFireballParticlePrefab);
            activeParticles.Add(particle);
            particle.transform.position = targetHand.transform.position + new Vector3(Random.Range(-0.05f,0.05f), Random.Range(-0.05f, 0.05f), Random.Range(-0.05f, 0.05f));
        }
    }

    private void ShootFireball()
    {
        foreach (var particle in activeParticles)
        {
            Destroy(particle);
        }
        activeParticles = new List<GameObject>();

        var fireball = Instantiate(fireballPrefab);
        fireball.transform.position = targetHand.transform.position;
        fireball.GetComponent<Fireball>().Shoot(Vector3.forward);
    }
}
